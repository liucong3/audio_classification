#from builtins import bytes
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SRUCell(nn.Module):
    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=1, use_relu=0):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout # horizontal on h+
        self.rnn_dropout = rnn_dropout # vertical on x+
        assert not (use_tanh and use_relu)
        self.use_tanh = use_tanh
        self.use_relu = use_relu
        self.dir_ = 2 if bidirectional else 1

        k = 4 if n_in != n_out * self.dir_ else 3
        self.weight = nn.Parameter(torch.Tensor(n_in, n_out * k * self.dir_))
        self.bias = nn.Parameter(torch.Tensor(n_out * 2 * self.dir_))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.n_in) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        self.bias.data.zero_().add_(bias_val)

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1-p).div_(1-p))

    def forward(self, x, c=None):
        # x.size() = (length, batch, n_in)
        input_dim = x.dim()
        assert input_dim == 2 or input_dim == 3
        batch = x.size(-2)
        if c is None:
            c = Variable(x.data.new(batch, self.n_out * self.dir_).zero_())

        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch, self.n_in), self.rnn_dropout)
            x = x * mask.expand_as(x)

        length = x.size(0) if x.dim() == 3 else 1
        x = x if x.dim() == 2 else x.contiguous().view(-1, self.n_in)
        # x.size() = (length * batch, n_in)
        u = x.mm(self.weight)
        # u.size() = (length * batch, n_out * k * dir_)

        mask_h = None
        if self.training and (self.dropout > 0):
            mask_h = self.get_dropout_mask_((batch, self.n_out * self.dir_), self.dropout)
        
        u = u.view(length, batch, -1, self.dir_, self.n_out)
        h, c = self.sru_compute(u, x, c, mask_h)
        if input_dim == 2:
            h.squeeze(0)
        return h, c

    def sru_compute(self, u, x, c0, mask_h):
        length, batch, k, dir_, n_out = u.size()
        assert dir_ == self.dir_ and n_out == self.n_out

        h = Variable(x.data.new(dir_, length, batch, n_out))
        c = Variable(x.data.new(dir_, batch, n_out))

        u = u.transpose(0, 2).transpose(1, 3) #.contiguous()
        # u.size() = (k, dir_, length, batch, n_out)
        if self.n_in != n_out * dir_:
            x = u[3]
        else:
            x = x.view(length, batch, dir_, n_out).transpose(0, 2).transpose(1, 2)
            # x.size() = (dir_, length, batch, n_out)
        gate = u[1:3] # gate.size() = (2, dir_, length, batch, n_out)
        bias = self.bias.view(2, dir_, n_out).unsqueeze(2).unsqueeze(2)
        gate = F.sigmoid(gate.add(bias.expand_as(gate)))
        gate_f = gate[0]
        gate_h = gate[1]
        gate = None

        u = u[0] * gate_f # u.size() = (dir_, length, batch, n_out)
        c_ = c0.view(batch, dir_, n_out).transpose(0, 1)
        # c.size() = (dir_, batch, n_out)

        def fwd(c1, dir_, i):
            c1 = c1 * (1 - gate_f[dir_, i]) + u[dir_, i]
            h[dir_, i] = c1
            return c1

        for d in range(self.dir_):
            c1 = c_[d]
            for t in range(length):
                i = t if d == 0 else length - 1 - t
                c1 = fwd(c1, d, i)
            c[d] = c1
   
        if self.use_tanh:
            h = F.tanh(h)
        if self.use_relu:
            h = F.relu(h)
        if not mask_h is None:
            mask_h = mask_h.view(dir_, batch, n_out).unsqueeze(1).expand_as(h)
            h = h * mask_h
        h = h * gate_h + x * (1 - gate_h)

        h = h.transpose(1, 2).transpose(0, 2).contiguous().view(length, batch, dir_ * n_out)
        c = c.transpose(0, 1).contiguous().view(batch, dir_ * n_out)
        return h, c


class SRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0, rnn_dropout=0,
                bidirectional=False, use_tanh=True, use_relu=False):
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.rnn_lst = nn.ModuleList()
        self.dir_ = 2 if bidirectional else 1

        for i in range(num_layers):
            l = SRUCell(
                n_in = self.n_in if i == 0 else self.n_out * self.dir_,
                n_out = self.n_out,
                dropout = dropout if i + 1 != num_layers else 0,
                rnn_dropout = rnn_dropout,
                bidirectional = bidirectional,
                use_tanh = use_tanh,
                use_relu = use_relu,
            )
            self.rnn_lst.append(l)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c=None, return_hidden=False):
        assert input.dim() == 3 # (length, batch, n_in)
        if c is None:
            zeros = Variable(input.data.new(
                input.size(1), self.n_out * self.dir_
            ).zero_())
            c = [ zeros for i in range(self.depth) ]
        else:
            assert c.dim() == 3    # (depth, batch, n_out*dir_)
            c = [ x.squeeze(0) for x in c.chunk(self.depth, 0) ]

        prevx = input
        for i, rnn in enumerate(self.rnn_lst):
            h, c[i] = rnn(prevx, c[i])
            prevx = h

        # sum the output of the two directions
        if self.dir_ == 2:
            time, batch, _ = prevx.size()
            prevx = prevx.view(time, batch, 2, self.n_out)
            prevx = prevx.sum(2).squeeze(2)

        if return_hidden:
            return prevx, torch.stack(c)
        else:
            return prevx


if __name__ == '__main__':

    cfg = {
        'input_size': 3, 
        'hidden_size': 4,
        'num_layers': 5, 
        'dropout': 0.3, 
        'rnn_dropout': 0.7, 
        'bidirectional': True,
        }
    sru = SRU(**cfg)
    length, batch, n_in = 7, 6, cfg['input_size']
    x = Variable(torch.randn(length, batch, n_in))
    h, c = sru(x, return_hidden=True)
    print h
    print c


