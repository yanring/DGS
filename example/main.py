import sys

import os

WORKPATH = os.path.abspath(os.path.dirname(os.path.dirname('main.py')))
sys.path.append(WORKPATH)
from distbelief.utils import messaging

from distbelief.utils.serialization import mp_gradient_filter

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from distbelief.optim import GradientSGD

from datetime import datetime
from example.models import AlexNet
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

import torch.optim as optim
from distbelief.server import GradientServer, GradientWarehouse


def get_dataset(args, transform):
    """
    :param dataset_name:
    :param transform:
    :param batch_size:
    :return: iterators for the dataset
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    sampler = DistributedSampler(trainset, args.world_size - 1, args.rank - 1)
    # sampler = DistributedSampler(trainset, 1, 0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                              sampler=sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def main(args):
    logs = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader, testloader = get_dataset(args, transform)
    net = AlexNet()
    # net = ResNet50()
    if args.cuda:
        net = net.cuda()
    # net.share_memory()
    # ResNet()

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0)
    else:
        print('distributed model')
        optimizer = GradientSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
        # optimizer = DownpourSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-5, cooldown=1,
                                                     factor=0.25)

    # train
    net.train()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Training for epoch {}".format(epoch))
        # set distributed_sampler.epoch to shuffle data.
        trainloader.sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            # optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            # _, a = ravel_sparse_gradient(net)
            # b = unravel_sparse_gradient(a)
            paralist = mp_gradient_filter(net)
            optimizer.step()
            for para1, para2 in zip(paralist, net.parameters()):
                para2.grad.data = para1
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(predicted, labels)

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': loss.item(),
                'training_accuracy': accuracy,
            }
            if i % 20 == 0:
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Accuracy : {training_accuracy:6.4f} | ".format(**log_obj))

            logs.append(log_obj)
        if True:  # print every n mini-batches
            logs[-1]['test_loss'], logs[-1]['test_accuracy'] = evaluate(net, testloader, args)
            print("Timestamp: {timestamp} | "
                  "Iteration: {iteration:6} | "
                  "Loss: {training_loss:6.4f} | "
                  "Accuracy : {training_accuracy:6.4f} | "
                  "Test Loss: {test_loss:6.4f} | "
                  "Test Accuracy: {test_accuracy:6.4f}".format(**logs[-1]))
        # val_loss, val_accuracy = evaluate(net, testloader, args, verbose=True)
        scheduler.step(logs[-1]['test_loss'])

    df = pd.DataFrame(logs)
    print(df)
    if args.no_distributed:
        if args.cuda:
            df.to_csv('log/gpu.csv', index_label='index')
        else:
            df.to_csv('log/single.csv', index_label='index')
    else:
        df.to_csv('log/node{}.csv'.format(dist.get_rank()), index_label='index')

    print('Finished Training')


def evaluate(net, testloader, args, verbose=False):
    if args.dataset == 'MNIST':
        classes = [str(i) for i in range(10)]
    else:
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net.eval()

    test_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += F.cross_entropy(outputs, labels).item()

    test_accuracy = accuracy_score(predicted, labels)
    if verbose:
        print('Loss: {:.3f}'.format(test_loss))
        print('Accuracy: {:.3f}'.format(test_accuracy))
        print(classification_report(predicted, labels, target_names=classes))

    return test_loss, test_accuracy


def init_server(args):
    os.system('rm *.size')
    model = AlexNet()
    if messaging.isCUDA:
        model.cuda()
    gradient_warehouse = GradientWarehouse(worker_num=args.world_size, model=model)
    threads_num = dist.get_world_size() - 1
    threads = []
    for i in range(1, threads_num + 1):
        th = GradientServer(model=model, gradient_warehouse=gradient_warehouse, rank=i)
        threads.append(th)
        th.start()
    for t in threads:
        t.join()
    # server = ParameterServer(model=model)
    # time.sleep(10000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--num-pull', type=int, default=5, metavar='N', help='how often to pull params (default: 5)')
    parser.add_argument('--num-push', type=int, default=5, metavar='N', help='how often to push grads (default: 5)')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA for training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how often to evaluate and print out')
    parser.add_argument('--no-distributed', action='store_true', default=False,
                        help='whether to use DownpourSGD or normal SGD')
    parser.add_argument('--rank', type=int, metavar='N',
                        help='rank of current process (0 is server, 1+ is training node)')
    parser.add_argument('--world-size', type=int, default=3, metavar='N', help='size of the world')
    parser.add_argument('--server', action='store_true', default=False, help='server node?')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='which dataset to train on')
    parser.add_argument('--master', type=str, default='localhost', help='ip address of the master (server) node')
    parser.add_argument('--port', type=str, default='29500', help='port on master node to communicate with')
    args = parser.parse_args()
    print(args)

    if not args.no_distributed:
        """ Initialize the distributed environment.
        Server and clients must call this as an entry point.
        """
        # os.environ['MASTER_ADDR'] = args.master
        # os.environ['MASTER_PORT'] = args.port
        print('%s/sharedfile chmod' % WORKPATH)
        if os.path.exists('%s/sharedfile' % WORKPATH):
            try:
                os.chmod('%s/sharedfile' % WORKPATH, 0o777)
                print('%s/sharedfile chmod success' % WORKPATH)
            except Exception as e:
                print(e)
        dist.init_process_group('tcp', init_method='file://%s/sharedfile' % WORKPATH, group_name='mygroup',
                                world_size=args.world_size, rank=args.rank)
        if args.cuda:
            messaging.isCUDA = 1
        if args.server:
            init_server(args)
    main(args)
