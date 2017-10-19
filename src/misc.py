# -*- coding: utf-8 -*-

def prepare_single_device(args):
    import torch
    torch.cuda.set_device(args.gpu)


class Average(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir(path, erase_old=False):
    import os, shutil
    if os.path.exists(path) and erase_old:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def datetimestr():
    def get_text(value):
        if value < 10:
            return '0%d' % value
        else:
            return '%d' % value
    import datetime
    now = datetime.datetime.now()
    return get_text(now.year) + get_text(now.month) + get_text(now.day) + '_' + \
            get_text(now.hour) + get_text(now.minute) + get_text(now.second)


def format_time(seconds, with_ms=False):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    if days > 0:
        f += str(days) + '/'
    if hours > 0:
        f += str(hours) + ':'
    f += str(minutes) + '.' + str(secondsf)
    if with_ms and millis > 0:
        f += '_' + str(millis)
    return f


def load_file(filename):
    import os, json
    if not os.path.isfile(filename):
        print('文件不存在 "%s"' % filename)
        return None
    with open(filename) as f:
        return f.read()


def save_file(filename, text):
    with open(filename, 'w') as f:
        return f.write(text)


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

