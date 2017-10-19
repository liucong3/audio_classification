import tqdm, fnmatch, random, io, subprocess
from tqdm import tqdm

def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)

def sort_files(data_files):
    data = []
    print 'getting file sizes...'
    for data_file in tqdm(data_files):
        data.append((data_file, get_audio_length(data_file)))
    data.sort(key=lambda x: x[1])
    print 'sorting...'
    return data

def create_manifest(manifest_file, data_files):
    data = sort_files(data_files)
    print 'dumping: "%s"...' % manifest_file
    with io.FileIO(manifest_file, "w") as file:
        for d in tqdm(data):
            label = int(d[0].split('/')[-1].split('-')[1])
            line = (d[0] + ',{},{}\n').format(label, d[1])
            file.write(line.encode('utf-8'))

if __name__ == '__main__':
    from train import parse_arguments
    import os, misc
    args = parse_arguments()

    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(args.target_dir)
                 for f in fnmatch.filter(files, '*.wav')]
    print 'Number of WAV files: {}'.format(len(wav_files))
    random.shuffle(wav_files)
    train_size = len(wav_files) - args.eval_size - args.test_size
    train_data = wav_files[0 : train_size]
    eval_data = wav_files[train_size : train_size + args.eval_size]
    test_data = wav_files[train_size + args.eval_size : ]
    misc.ensure_dir(args.manifest_dir, erase_old=True)
    create_manifest(os.path.join(args.manifest_dir, 'train.csv'), train_data)
    create_manifest(os.path.join(args.manifest_dir, 'eval.csv'), eval_data)
    create_manifest(os.path.join(args.manifest_dir, 'test.csv'), test_data)
