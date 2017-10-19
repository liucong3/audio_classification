from __future__ import print_function
import os, torch, misc

def load_audio(audio_path):
    import torchaudio
    sound, sample_rate = torchaudio.load(audio_path)
    assert sound.dim() == 1 or sound.dim() == 2
    if sound.dim() > 1:
        sound = sound.mean(1)
        if sound.dim() > 1:
            sound = sound.squeeze(1)
    return sound, sample_rate  

def audio_spectrogram(sound, sample_rate, window_size, window_stride):
    import librosa, scipy
    sound = sound.numpy()
    win_length = int(sample_rate * window_size)
    # print 'win_length', win_length, 'hop_length', 1 + sound.shape[0] / int(sample_rate * window_stride)
    stft = librosa.stft(sound, 
        n_fft=win_length, 
        hop_length=int(sample_rate * window_stride), 
        win_length=win_length, 
        window=scipy.signal.hamming)
    spectrogram, phase = librosa.magphase(stft)
    spectrogram = torch.FloatTensor(spectrogram)
    spectrogram = spectrogram.log1p() # S = log(S+1)
    return spectrogram # spectrogram.size() = (#spectrum, hops)

def transfer_folder(in_folder, out_folder, args):
    for file in os.listdir(in_folder):
        in_file = os.path.join(in_folder, file)
        if not os.path.isfile(in_file): continue
        out_file = os.path.join(out_folder, file)
        sound, sample_rate = load_audio(in_file)
        spectrogram = audio_spectrogram(sound, args.sample_rate, args.window_size, args.window_stride)
        torch.save(spectrogram, out_file)

def transfer(data_16000_dir, data_dir, args):
    misc.ensure_dir(data_dir, erase_old=True)

    for file in os.listdir(data_16000_dir):
        in_folder = os.path.join(data_16000_dir, file)
        if not os.path.isdir(in_folder): continue
        out_folder = os.path.join(data_dir, file + '.pth')
        misc.ensure_dir(out_folder, erase_old=True)
        transfer_folder(in_folder, out_folder, args)

if __name__ == '__main__':    
    from train import parse_arguments
    args = parse_arguments()

    proceed = 'y'
    if not os.path.exists(args.data_16000_dir):
        print('Data folder "{}" does not exist.'.format(args.data_16000_dir))
        proceed = 'n'
    if proceed == 'y' and os.path.exists(args.data_dir):
        prompt = 'To proceed, everything in folder "{}" will be erased. [y/N]?'.format(args.data_dir)
        proceed = raw_input(prompt)
    if proceed == 'y':
        transfer(args.data_16000_dir, args.data_dir, args)

