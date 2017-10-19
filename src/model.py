import math, torch, torch.nn as nn, torch.nn.functional as F

class AudioModel(nn.Module):

    def __init__(self, rnn_type, bidirectional=True, config=None):
        super(AudioModel, self).__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        if config is not None:
            self.setup(config)

    def setup(self, config):
        self.config = config
        if 'rnn_type' not in config:
            config['rnn_type'] = self.rnn_type.__class__.__name__
        assert config['rnn_type'] == self.rnn_type.__class__.__name__
        sample_rate = config['sample_rate']
        window_size = config['window_size']
        hidden_size = config['hidden_size']
        hidden_layers = config['hidden_layers']
        num_classes = config['num_classes']

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P) / S + 1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1) # num of spectrums
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32 # num of channels in the last CNN
        self.rnn_layers = self.rnn_type(rnn_input_size, hidden_size, hidden_layers, bidirectional=self.bidirectional)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes, bias=False)
        )

    def forward(self, x):
        # x.size() = (batch, 1, freq, time)
        x = self.conv(x) # x.size() = (batch, 32, input_size, time)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # NxHxT -> TxNxH

        x = self.rnn_layers(x)
        x = x.sum(0).squeeze(0) # TxNxH -> NxH

        x = self.fc(x)
        return x

    def serialize(self):
        assert 'rnn_type' in self.config
        package = {
            'config': self.config,
            'state_dict': self.state_dict(),
        }
        return package

    def save(self, path):
        torch.save(self.serialize(), path)

    def load(self, path):
        package = torch.load(path, map_location=lambda storage, location: storage)
        self.setup(package['config'])
        self.load_state_dict(package['state_dict'])
        return package['config']


def cross_entropy_loss(pred, truth):
    return torch.nn.functional.cross_entropy(pred, truth)


if __name__ == '__main__':
    print 'TEST'
    from train import parse_arguments
    import rnn, sru
    args = parse_arguments()
    # model = AudioModel(rnn.BatchRNNLayers, config=args.__dict__)
    model = AudioModel(sru.SRU, config=args.__dict__)
     
    import loader
    from torch.autograd import Variable
    manifest_path = 'manifest/test.csv'
    dataloader, _, _ = loader.get(args, manifest_path)
    for input, target in dataloader:
        print '- forward'
        output = model(Variable(input))
        loss = cross_entropy_loss(output, Variable(target))
        print '- backward'
        loss.backward()
        break

    print '- save'
    model.save(args.model_path)
    print '- load'
    model.load(args.model_path)

