import argparse
import os
import socket
import sys
import time

WORKPATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKPATH)
from torch.optim.lr_scheduler import MultiStepLR

from core.optim import GradientSGD
from core.utils import constant
from core.utils.GradualWarmupScheduler import GradualWarmupScheduler
from core.utils.serialization import ravel_model_params
from example.main import init_server

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime
from example.models import *
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


def get_dataset(args, transform_train, transform_test):
    """
    :param args:
    :param transform:
    :return:
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='%s/data' % WORKPATH, train=True, download=True,
                                              transform=transform_train)
        testset = torchvision.datasets.MNIST(root='%s/data' % WORKPATH, train=False, download=True,
                                             transform=transform_test)
    elif args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='%s/data' % WORKPATH, train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='%s/data' % WORKPATH, train=False, download=True,
                                               transform=transform_test)

    sampler = DistributedSampler(trainset, args.world_size - 1, args.rank - 1)
    # sampler = DistributedSampler(trainset, 1, 0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                              sampler=sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def init_net(args):
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
    if args.cuda:
        net = net.cuda()
    if args.no_distributed and args.half:
        net = net.half()
    return net


def cifar10(args, optimizer, net):
    logs = []
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainloader, testloader = get_dataset(args, transform_train, transform_test)

    if args.warmup:
        args.lr = args.lr / 10

        # optimizer = DownpourSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, cooldown=1, verbose=True, factor=0.25)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    if args.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=4,
                                           after_scheduler=scheduler)
    compress_ratio = [0.01] * (args.epochs + 10)
    compress_ratio[1:4] = [0.25, 0.0625, 0.0625 * 0.25, 0.01]
    # train
    net.train()

    for epoch in range(1, args.epochs + 1):  # loop over the dataset multiple times
        # scheduler.step()
        if args.no_distributed or args.rank == 1:
            # scheduler.step(logs[-1]['test_loss'])
            scheduler.step(epoch)
        if not args.no_distributed:
            optimizer.compress_ratio = compress_ratio[epoch]
        print("Training for epoch {}, lr={}".format(epoch, scheduler.optimizer.param_groups[0]['lr']))
        net.train()
        # set distributed_sampler.epoch to shuffle data.
        trainloader.sampler.set_epoch(epoch)
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            if args.no_distributed and args.half:
                inputs = inputs.half()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(predicted.cpu(), labels.cpu())

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.item(),
                'training_accuracy': accuracy,
            }
            if i % 80 == 0:
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Accuracy : {training_accuracy:6.4f} | ".format(**log_obj))

            logs.append(log_obj)
        if True:  # print every n mini-batches
            end = time.time()
            print('minibatch cost :%f, time cost: %f' % ((end - start) / (781 / (args.world_size - 1)), (end - start)))
            logs[-1]['test_loss'], logs[-1]['test_accuracy'] = evaluate(net, testloader, args)
            print("Timestamp: {timestamp} | "
                  "Iteration: {iteration:6} | "
                  "Loss: {training_loss:6.4f} | "
                  "Accuracy : {training_accuracy:6.4f} | "
                  "Test Loss: {test_loss:6.4f} | "
                  "Test Accuracy: {test_accuracy:6.4f}".format(**logs[-1])
                  )
        # val_loss, val_accuracy = evaluate(net, testloader, args, verbose=True)

        df = pd.DataFrame(logs)
        with open(WORKPATH + '/running.log', 'a+') as f:
            running_log = '{},node{}_{}_{}_m{}_e{}_{}.csv'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                                  args.rank - 1, args.mode,
                                                                  args.model, args.momentum,
                                                                  epoch,
                                                                  logs[-1]['test_accuracy'])
            f.write(running_log + '\n')

    print(df)
    if args.no_distributed:
        if args.cuda:
            df.to_csv(
                WORKPATH + '/log/gpu_{}_{}_m{}_e{}_b{}_{}.csv'.format(args.mode, args.model, args.momentum, args.epochs,
                                                                      args.batch_size, logs[-1]['test_accuracy']),
                index_label='index')
        else:
            df.to_csv(WORKPATH + '/log/single.csv', index_label='index')
    else:
        df.to_csv(WORKPATH + '/log/node{}_{}_{}_m{}_e{}_b{}_{}worker_dual_{}.csv'.format(args.rank - 1, args.mode,
                                                                                         args.model, args.momentum,
                                                                                         args.epochs,
                                                                                         args.batch_size,
                                                                                         args.world_size - 1,
                                                                                         logs[-1]['test_accuracy']),
                  index_label='index')
    print('Finished Training')


def evaluate(net, testloader, args, verbose=False):
    if args.dataset == 'MNIST':
        classes = [str(i) for i in range(10)]
    else:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            if args.no_distributed and args.half:
                images = images.half()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += F.cross_entropy(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum()

    fake_test_accuracy = accuracy_score(predicted, labels)
    test_accuracy = correct.item() / total
    print('%f,%f,%f|%f,%s' % (
        test_accuracy, correct.item(), total, fake_test_accuracy, str((predicted == labels).sum())))
    if verbose:
        print('Loss: {:.3f}'.format(test_loss))
        print('Accuracy: {:.3f}'.format(test_accuracy))
        print(classification_report(predicted, labels, target_names=classes))

    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')

    parser.add_argument('--test-batch-size', type=int, default=20000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='momentum', help='momentum (default: 0.0)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)',
                        dest='weight_decay')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA for training')
    parser.add_argument('--warmup', action='store_true', default=False, help='use warmup or not')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False,
                        help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, default=0, metavar='N',
                        help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    # parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to train on')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='29500', help='port on master node to communicate with')
    parser.add_argument('--mode', type=str, default='gradient_sgd', help='gradient_sgd, dgc, Aji or asgd')
    parser.add_argument('--model', type=str, default='ResNet18', help='AlexNet, ResNet18, ResNet50')
    parser.add_argument('--network-interface', type=str, default=None,
                        help='By default, Gloo backends will try to find the right network interface to use. '
                             'If the automatically detected interface is not correct, you can override it ')
    args = parser.parse_args()
    if args.network_interface:
        os.environ['GLOO_SOCKET_IFNAME'] = args.network_interface
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
        args.warmup = False
        net = init_net(args)
        print('MODEL:%s, momentum:%f' % (args.model, args.momentum))
        assert net is not None
        constant.MODEL_SIZE = ravel_model_params(net).numel()
        if args.rank == 0 and not args.no_distributed:
            if args.cuda is False:
                print('server init in cpu')
            init_server(args, net)
        else:
            optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    args=args)
            cifar10(args, optimizer, net)
