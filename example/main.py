import os
import socket
import sys

from torch import optim

WORKPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKPATH)
sys.path.append(WORKPATH)
from distbelief.optim import GradientSGD

from distbelief.utils.serialization import ravel_model_params

from distbelief.utils import constant

import argparse
import torch.distributed as dist
from example.models import *
from distbelief.server import GradientServer


def init_server(args, net):
    print('init server!!!')
    dist.init_process_group('tcp', init_method='file://%s/sharedfile' % WORKPATH, group_name='mygroup',
                            world_size=args.world_size, rank=args.rank)

    if args.cuda:
        model = net.cuda()
    else:
        model = net
    size_list = [i.data.numel() for i in net.parameters()]
    threads_num = dist.get_world_size() - 1
    threads = []
    global_model = ravel_model_params(model, cuda=True)
    constant.MODEL_SIZE = global_model.numel()
    synced_model = global_model.clone()
    for i in range(1, threads_num + 1):
        th = GradientServer(model=model, rank=i, worker_num=args.world_size, global_model=global_model,
                            synced_model=synced_model, size_list=size_list)
        threads.append(th)
        th.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=20000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='momentum', help='momentum (default: 0.0)')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA for training')
    parser.add_argument('--warmup', action='store_true', default=False, help='use warmup or not')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False,
                        help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, metavar='N',
                        help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    # parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='29500', help='port on master node to communicate with')
    parser.add_argument('--mode', type=str, default='gradient_sgd', help='gradient_sgd, dgc, Aji or asgd')
    parser.add_argument('--model', type=str, default='AlexNet', help='AlexNet, ResNet18, ResNet50')
    args = parser.parse_args()
    if args.dataset == 'an4':
        # parser = argparse.ArgumentParser(description='DeepSpeech Training With AN4')
        parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
        parser.add_argument('--num-workers', default=2, type=int, help='Number of workers used in data-loading')
        parser.add_argument('--labels-path', default=WORKPATH + '/deepspeech/labels.json',
                            help='Contains all characters for transcription')
        parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
        parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
        parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
        parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
        parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
        parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
        parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
        parser.add_argument('--learning-anneal', default=1 / 1.1, type=float,
                            help='Annealing applied to learning rate every epoch')
        parser.add_argument('--silent', dest='silent', action='store_true',
                            help='Turn off progress tracking per iteration')
        parser.add_argument('--noise-dir', default=None,
                            help='Directory to inject noise into audio. If default, noise Inject not added')
        parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
        parser.add_argument('--noise-min', default=0.0,
                            help='Minimum noise level to sample from. (1.0 means all noise, not original signal)',
                            type=float)
        parser.add_argument('--noise-max', default=0.5, help='Maximum noise levels to sample from. Maximum 1.0',
                            type=float)
        parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                            help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
        parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                            help='Turn off ordering of dataset on sequence length for the first epoch.')
        parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                            help='Turn off bi-directional RNNs, introduces lookahead convolution')

        parser.add_argument('--train-manifest', metavar='DIR', help='path to train manifest csv',
                            default=WORKPATH + '/data/an4_train_manifest.csv')
        parser.add_argument('--val-manifest', metavar='DIR', help='path to validation manifest csv',
                            default=WORKPATH + '/data/an4_val_manifest.csv')

    args = parser.parse_args()

    if args.cuda:
        if socket.gethostname() == 'yan-pc':
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 1)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 2)
            # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print('Using device%s, device count:%d' % (os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count()))

    # args.model = 'ResNet50'
    args.momentum = 0.7
    args.half = False
    args.weight_decay = 5e-4
    # args.warmup = True
    net = None
    if args.dataset == 'an4':
        from example.an4 import init_net

        args.lr = 5e-4
        args.epochs = 100
        args.batch_size = 20
        args.weight_decay = 0
        args.momentum = 0.9
        net = init_net(args)
    elif args.dataset == 'cifar10':
        from example.cifar10 import cifar10, init_net

        args.model = 'ResNet18'
        net = init_net(args)
    print('MODEL:%s, momentum:%f' % (args.model, args.momentum))
    print(args)
    # print(net)
    assert net is not None
    constant.MODEL_SIZE = ravel_model_params(net).numel()
    if args.rank == 1 and args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)
    else:
        print('distributed model')
        optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum,
                                weight_decay=args.weight_decay * (args.world_size - 1) / 4,
                                # weight_decay=5e-6,
                                args=args)
    # if not args.no_distributed:
    #     """ Initialize the distributed environment.
    #     Server and clients must call this as an entry point.
    #     """
    #     print('%s/sharedfile chmod' % WORKPATH)
    #     if os.path.exists('%s/sharedfile' % WORKPATH):
    #         try:
    #             os.chmod('%s/sharedfile' % WORKPATH, 0o777)
    #             print('%s/sharedfile chmod success' % WORKPATH)
    #         except Exception as e:
    #             print(e)
    #     if args.rank == 0:
    #         import glob
    #
    #         for infile in glob.glob(os.path.join(WORKPATH, '*.size')):
    #             os.remove(infile)
    if args.dataset == 'an4':
        from example.an4 import an4

        if args.rank == 0 and not args.no_distributed:
            init_server(args, net)
        else:
            an4(args, optimizer, net)

    elif args.dataset == 'cifar10':
        if args.rank == 0 and not args.no_distributed:
            init_server(args, net)
        else:
            cifar10(args, optimizer, net)
