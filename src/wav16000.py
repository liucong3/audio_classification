import os

def transfer_folder(in_folder, out_folder, sample_rate):
    for file in os.listdir(in_folder):
        in_file = os.path.join(in_folder, file)
        if not os.path.isfile(in_file): continue
        out_file = os.path.join(out_folder, file)    
        # -t File type of audio
        # -r Sample rate of audio
        # -b Encoded sample size in bits
        # -e Set encoding
        # -B Big endian
        # -c Number of channels of audio data
        # cmd = 'sox -t wav -r %d -b 16 -e signed-integer -B -c 1 \"%s\" \"%s\"' % (sample_rate, in_file, out_file)
        cmd = 'sox \"%s\" -r %d -b 16 -c 1 \"%s\"' % (in_file, sample_rate, out_file)
        os.system(cmd)

def transfer(raw_data_dir, data_dir, sample_rate):
    import misc
    misc.ensure_dir(data_dir, erase_old=True)

    for file in os.listdir(raw_data_dir):
        in_folder = os.path.join(raw_data_dir, file)
        if not os.path.isdir(in_folder): continue
        out_folder = os.path.join(data_dir, file)
        misc.ensure_dir(out_folder, erase_old=True)
        transfer_folder(in_folder, out_folder, sample_rate)

if __name__ == '__main__':    
    from train import parse_arguments
    args = parse_arguments()

    proceed = 'y'
    if not os.path.exists(args.raw_data_dir):
        print 'Raw data folder "{}" does not exist.'.format(args.raw_data_dir)
        proceed = 'n'
    if proceed == 'y' and os.path.exists(args.data_16000_dir):
        prompt = 'To proceed, everything in folder "{}" will be erased. [y/N]?'.format(args.data_16000_dir)
        proceed = raw_input(prompt)
    if proceed == 'y':
        transfer(args.raw_data_dir, args.data_16000_dir, args.sample_rate)


