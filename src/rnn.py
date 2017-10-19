import torch, torch.nn as nn
from collections import OrderedDict

class SequenceWise(nn.Module):

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type=nn.GRU, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=False)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        if hasattr(self.rnn, 'flatten_parameters'):
            self.rnn.flatten_parameters()
        return x

class BatchRNNLayers(nn.Sequential):

    def __init__(self, input_size, hidden_size, hidden_layers, rnn_type=nn.GRU, bidirectional=False):
        rnns = []
        for x in range(hidden_layers):
            rnn = BatchRNN(input_size=input_size, hidden_size=hidden_size, 
                            rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=(x>0))
            rnns.append(('%d' % x, rnn))
            input_size = hidden_size
        super(BatchRNNLayers, self).__init__(OrderedDict(rnns))
