import os
import socket
import sys

import torchvision

WORKPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKPATH)
sys.path.append(WORKPATH)
from distbelief.optim import GradientSGD
from distbelief.utils.log import Log

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
    global_model = ravel_model_params(model)
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

    parser.add_argument('--test-batch-size', type=int, default=20000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
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
    if args.cuda:
        if socket.gethostname() == 'yan-pc':
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 1)
        elif 'gn' in socket.gethostname():
            print('init in th')
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 4)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 2)
            # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print('Using device%s, device count:%d' % (os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count()))

    net = None

    if args.dataset == 'cifar10':
        from example.cifar10 import cifar10, init_net

        args.model = 'ResNet18'
        args.momentum = 0.7
        args.half = False
        args.weight_decay = 5e-4
        args.warmup = False
        net = init_net(args)
        print('MODEL:%s, momentum:%f' % (args.model, args.momentum))
        print(args)
        assert net is not None
        constant.MODEL_SIZE = ravel_model_params(net).numel()
        if args.rank == 0 and not args.no_distributed:
            if args.world_size > 40:
                args.cuda = False
                net = net.cpu()
                print('server init in cpu')
            init_server(args, net)
        else:
            optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    # weight_decay=5e-6,
                                    args=args)
            try:
                cifar10(args, optimizer, net)
            except Exception:
                err_logger = Log('error', cmdlevel='ERROR', filename='err.log', backup_count=1, when='D')
                err_logger.trace()
    elif args.dataset == 'tinyimagenet':

        args.model = 'resnet18'
        args.momentum = 0.9
        args.half = False
        args.weight_decay = 1e-4
        args.batch_size = 256
        args.epochs = 90
        # args.warmup = True
        net = torchvision.models.resnet18(num_classes=200).cuda()
        print('MODEL:%s, momentum:%f' % (args.model, args.momentum))
        print(args)
        assert net is not None
        constant.MODEL_SIZE = ravel_model_params(net).numel()
        if args.rank == 0 and not args.no_distributed:
            if args.world_size > 17:
                args.cuda = False
            init_server(args, net)
        else:
            optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    # weight_decay=5e-6,
                                    args=args)
            try:
                from example.Imagenet_Origin import main

                main(parser, optimizer)
            except Exception as e:
                err_logger = Log('error', cmdlevel='ERROR', filename='err.log', backup_count=1, when='D')
                err_logger.trace()
                with open(WORKPATH + '/err.log', 'a+') as f:
                    running_log = str(e)
                    f.write(running_log + '\n')
