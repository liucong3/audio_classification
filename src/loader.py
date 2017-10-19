import torch, numpy
from collections import defaultdict 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

def normalize(data):
    mean, std = data.mean(), data.std()
    return data.add_(-mean).div_(std)

def get_manifest(manifest_path):
    with open(manifest_path) as file:
        return [line.strip().split(',') for line in file.readlines()]

class SpectrogramDataset(Dataset):

    # def __init__(self, manifest_path, args, transform=None):
    def __init__(self, manifest_path, args):
        super(SpectrogramDataset, self).__init__()
        self.window_size = args.window_size
        self.window_stride = args.window_stride
        self.samples = get_manifest(manifest_path)
        # self.transform = transform

    def __getitem__(self, index):
        audio_path, target, size = self.samples[index]
        spectrogram = torch.load(audio_path)
        # if self.transform is not None:
        #     spectrogram = self.transform(spectrogram)
        spectrogram = normalize(spectrogram)
        return spectrogram, int(target)

    def __len__(self):
        return len(self.samples)


def spectrogram_dataset_collate_fn(batch):
    sample, _ = max(batch, key=lambda x: x[0].size(1))
    inputs = torch.zeros(len(batch), 1, sample.size(0), sample.size(1))
    targets = torch.LongTensor(len(batch))
    for idx, (spectrogram, target) in enumerate(batch):
        inputs[idx][0].narrow(1, 0, spectrogram.size(1)).copy_(spectrogram)
        targets[idx] = target
    return inputs, targets


class BucketingSampler(Sampler):

    def __init__(self, manifest_path):
        samples = get_manifest(manifest_path)
        audio_lengths = [float(sample[2]) for sample in samples]
        hist, bin_edges = numpy.histogram(audio_lengths, bins="auto")
        audio_samples_indices = numpy.digitize(audio_lengths, bins=bin_edges)
        self.bins_to_samples = defaultdict(list)
        for idx, bin_id in enumerate(audio_samples_indices):
            self.bins_to_samples[bin_id].append(idx)
        self.size = len(samples)

    def __iter__(self):
        for bin, sample_idx in self.bins_to_samples.items():
            numpy.random.shuffle(sample_idx)
            for s in sample_idx:
                yield s

    def __len__(self):
        return self.size

  
class RandPadTrim(object):

    def __init__(self, length, fill_value=0):
        self.length = length
        self.fill_value = fill_value

    def __call__(self, tensor):
        import numpy
        res = torch.ones(self.length) * self.fill_value
        res = res.type_as(res)
        if self.length == tensor.size(0):
            return tensor
        elif self.length > tensor.size(0):
            res = torch.zeros(self.length).type_as(tensor)
            res.fill_(self.fill_value)
            start = numpy.random.randint(0, self.length - tensor.size(0))
            res[start:start+tensor.size(0)].copy_(tensor)
            return res
        else: # self.length < tensor.size(0):
            start = numpy.random.randint(0, tensor.size(0) - self.length)
            return tensor[start:start+self.length]


def get(args, manifest_path=None, dataset=None, bucketing_sampler=None, no_bucketing=False):
    if dataset is None:
        assert manifest_path is not None
        # crop_audio_length = int(args.crop_audio_length * args.sample_rate)
        # dataset = SpectrogramDataset(manifest_path, args, transform=RandPadTrim(crop_audio_length))
        dataset = SpectrogramDataset(manifest_path, args)
    if no_bucketing:
        bucketing_sampler = None
    elif bucketing_sampler is None:
        assert manifest_path is not None
        bucketing_sampler = BucketingSampler(manifest_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                            collate_fn=spectrogram_dataset_collate_fn, sampler=bucketing_sampler)
    return dataloader, dataset, bucketing_sampler


if __name__ == '__main__':
    # TEST
    from train import parse_arguments
    args = parse_arguments()

    manifest_path = 'manifest/test.csv'
    dataloader1, _, _ = get(args, manifest_path, no_bucketing=True)
    dataloader2, _, _ = get(args, manifest_path)

    for input, target in dataloader1:
        print input.size()
    for input, target in dataloader2:
        print input.size()

 
