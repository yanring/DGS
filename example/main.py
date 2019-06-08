import sys

import os
import socket

WORKPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKPATH)
sys.path.append(WORKPATH)
from example.cifar10 import cifar10

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
    threads_num = dist.get_world_size() - 1
    threads = []
    global_model = ravel_model_params(model, cuda=True)
    constant.MODEL_SIZE = global_model.numel()
    synced_model = global_model.clone()
    for i in range(1, threads_num + 1):
        th = GradientServer(model=model, rank=i, worker_num=args.world_size, global_model=global_model,
                            synced_model=synced_model)
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
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
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

    if args.cuda:
        if socket.gethostname() == 'yan-pc':
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 1)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 2)
        print('Using device%s, device count:%d' % (os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count()))

    args.model = 'ResNet50'
    args.momentum = 0.7
    args.half = 'False'
    # args.warmup = True
    args.mode = 'gradient_sgd'
    print('MODEL:%s, momentum:%f' % (args.model, args.momentum))
    if args.model == 'AlexNet':
        net = AlexNet()
    elif args.model == 'ResNet18':
        net = ResNet18()
        args.test_batch_size = 1000
    elif args.model == 'ResNet50':
        net = ResNet50()
        args.test_batch_size = 1000
    elif args.model == 'ResNet101':
        net = ResNet101()
        args.test_batch_size = 500
    else:
        net = None
    print(args)
    if not args.no_distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        print('%s/sharedfile chmod' % WORKPATH)
        if os.path.exists('%s/sharedfile' % WORKPATH):
            try:
                os.chmod('%s/sharedfile' % WORKPATH, 0o777)
                print('%s/sharedfile chmod success' % WORKPATH)
            except Exception as e:
                print(e)
        if args.rank == 0:
            import glob

            for infile in glob.glob(os.path.join(WORKPATH, '*.size')):
                os.remove(infile)

        if args.rank == 0:
            init_server(args, net)

    cifar10(args)
