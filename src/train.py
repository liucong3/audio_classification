import os, torch, misc

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='DeepSpeech training')
    # wav16000
    parser.add_argument('--raw_data_dir', default='raw_data', help='Path to read raw dataset')
    parser.add_argument('--data_16000_dir', default='wav_16000', help='Path to save dataset')
    parser.add_argument('--data_dir', default='data', help='Path to save dataset')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
    # manifest
    parser.add_argument('--manifest_dir', default='manifest', help='Path to manifest')
    parser.add_argument('--eval_size', type=int, default=500, help='Size of held-out evaluation set')
    parser.add_argument('--test_size', type=int, default=500, help='Size of held-out test set')
    # data loader
    parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
    parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
    parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
    # parser.add_argument('--crop_audio_length', default=20, type=float, help='Crop audios exceeding this length in seconds')
    # audio model
    parser.add_argument('--hidden_size', default=800, type=int, help='Hidden size of RNNs')
    parser.add_argument('--hidden_layers', default=5, type=int, help='Number of RNN layers')
    parser.add_argument('--num_classes', default=10, type=int, help='Number of audio classes')
    parser.add_argument('--model_path', default='models/best.pth', help='Location to save best validation model')
    # train
    parser.add_argument('--gpu', default=-1, type=int, help='The single device used in training, -1 means using CPU or ALL GPUs if avaiable.')
    parser.add_argument('--train_manifest', default='train.csv', help='path to train manifest csv')
    parser.add_argument('--eval_manifest', default='eval.csv', help='path to eval manifest csv')
    parser.add_argument('--test_manifest', default='test.csv', help='path to test manifest csv')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
    parser.add_argument('--learning_anneal', default=1.05, type=float, help='Annealing applied to learning rate every epoch')
    parser.add_argument('--logdir', default='log', type=str, help='Log folder')
    parser.add_argument('--continue_from', type=str, help='Continue from last training epoch')
    parser.add_argument('--plot', action='store_true', help='Plot training progress in terms of accuracies')
    # parse
    return parser.parse_args()


class Logger(object):

    def __init__(self, args, init_train_info={}, sub_dir=None):
        self.args = args        
        misc.ensure_dir(args.logdir)
        sub_dir = args.continue_from or sub_dir or misc.datetimestr()
        self.logdir = os.path.join(args.logdir, sub_dir)
        misc.ensure_dir(self.logdir)
        self._setup_log_file()
        self._create_train_info(args, init_train_info)

    def _setup_log_file(self):
        import logging
        log_file = os.path.join(self.logdir, 'log.txt')
        self._logger = logging.getLogger()
        self._logger.setLevel('INFO')
        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)

    def info(self, str_info):
        self._logger.info(str_info)

    # train_info

    def _create_train_info(self, args, init_train_info):
        import json
        train_info_path = os.path.join(self.logdir, 'train_info.txt')
        if os.path.isfile(train_info_path):
            self.train_info = json.loads(misc.load_file(train_info_path))
        else:
            self.train_info = init_train_info

    def save_train_info(self):
        import json
        train_info_path = os.path.join(self.logdir, 'train_info.txt')
        misc.save_file(train_info_path, json.dumps(self.train_info))

    def plot_progress(self):
        import plot
        pdf_plot_path = os.path.join(self.logdir, 'progress.pdf')
        x_data = torch.Tensor(range(1, 1 + len(self.train_info['train_acc'])))
        y_data = torch.Tensor([self.train_info['train_acc'], self.train_info['eval_acc']])
        # y_data, y_err = plot.smooth2d(y_data, step=3)
        plot.plot(x_data, y_data, y_err=None, legends=['Train', 'Eval'], 
                title=None, xlabel='Epoch', ylabel='Acccuracy', filename=pdf_plot_path)

    # model

    def save_model(self, model, model_path=None):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        if model_path is None:
            if 'model_path' in self.train_info and os.path.exists(self.train_info['model_path']):
                print("Removing old model {}".format(self.train_info['model_path']))
                os.remove(os.join(self.logdir, self.train_info['model_path']))
            model_path = misc.datetimestr() + '.model.pth'
            self.train_info['model_path'] = model_path
        print("Saving model to {}".format(model_path))
        model.save(os.path.join(self.logdir, model_path))

    def create_model(self):
        import model, rnn
        if 'model_path' in self.train_info and os.path.exists(self.train_info['model_path']):
            model = model.AudioModel(rnn.BatchRNNLayers, config=None)
            model.load(self.train_info['model_path'])
        else:
            model = model.AudioModel(rnn.BatchRNNLayers, config=self.args.__dict__)
        if torch.cuda.is_available():
            if self.args.gpu == -1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        return model


def run_epoch(model, data_loader, is_cuda, optimizer=None, max_norm=None):
    import time
    from tqdm import tqdm
    from torch.autograd import Variable

    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    batch_time = misc.Average()
    avg_loss = misc.Average()
    avg_acc = misc.Average()

    end = time.time()
    for i, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        if is_train:
            input = Variable(input, requires_grad=False)
        else:
            input = Variable(input, volatile=True)            
        if is_cuda:
            input = input.cuda()

        output = model(input)
        _, pred = output.data.cpu().max(1)
        batch_size = input.size(0)
        acc = (pred.view(-1) == target).sum() / float(batch_size)
        avg_acc.update(acc, batch_size)

        if is_train:
            target = Variable(target, requires_grad=False)
            if is_cuda:
                target = target.cuda()

            loss = torch.nn.functional.cross_entropy(output, target) / batch_size
            avg_loss.update(loss.data[0], batch_size)
            optimizer.zero_grad()
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
            optimizer.step()

        if is_cuda:
            torch.cuda.synchronize()

        batch_time.update(time.time() - end)
        end = time.time()

    return avg_acc.avg, batch_time.avg, avg_loss.avg


def init_train_info():
    train_info = {}
    train_info['epoch'] = 0
    train_info['train_loss'] = []
    train_info['train_acc'] = []
    train_info['eval_acc'] = []
    train_info['test_acc'] = []
    train_info['best_eval_acc'] = 0
    train_info['best_test_acc'] = 0
    return train_info

def anneal_lr(epoch, args, optimizer, logger):
    if epoch > 1:
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        logger.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

def train(args):
    is_cuda = args.gpu >= 0
    logger = Logger(args, init_train_info=init_train_info())

    model = logger.create_model()
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, nesterov=True)
    logger.info("Number of parameters: %d" % misc.get_param_size(model))

    import loader
    train_loader, _, _ = loader.get(args, os.path.join(args.manifest_dir, args.train_manifest))
    eval_loader, _, _ = loader.get(args, os.path.join(args.manifest_dir, args.eval_manifest))
    test_loader, _, _ = loader.get(args, os.path.join(args.manifest_dir, args.test_manifest))

    for epoch in range(args.epochs):
        anneal_lr(epoch + 1, args, optimizer, logger)
        if epoch < logger.train_info['epoch']: continue
        logger.train_info['epoch'] += 1

        model.train()
        acc, batch_time, loss = run_epoch(model, train_loader, is_cuda, optimizer, args.max_norm)
        logger.info('Epoch: {}, train_acc: {}, train_loss: {}, train_batch_time: {}'.format(epoch + 1, acc, loss, batch_time))
        logger.train_info['train_loss'].append(loss)
        logger.train_info['train_acc'].append(acc)

        model.eval()
        acc, batch_time, _ = run_epoch(model, eval_loader, is_cuda)
        logger.info('Epoch: {}, eval_acc: {}, eval_batch_time: {}'.format(epoch + 1, acc, batch_time))
        logger.train_info['eval_acc'].append(acc)

        if acc > logger.train_info['best_eval_acc']:
            logger.train_info['best_eval_acc'] = acc
            acc, batch_time, _ = run_epoch(model, test_loader, is_cuda)
            logger.info('Epoch: {}, test_acc: {}, test_batch_time: {}'.format(epoch + 1, acc, batch_time))
            logger.train_info['test_acc'].append(acc)

            if acc > logger.train_info['best_test_acc']:
                print("Found better model (acc=%.5f) with test dataset." % acc)
                logger.train_info['best_test_acc'] = acc
                logger.save_model(model, 'best.model.pth')

        logger.save_train_info()
        logger.save_model(model)
        if args.plot:
            logger.plot_progress()


if __name__ == '__main__':
    args = parse_arguments()
    if args.gpu >= 0:
        misc.prepare_single_device(args)
    train(args)
