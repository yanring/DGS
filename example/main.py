import sys
import time

import os
import socket
from multiprocessing import Manager

WORKPATH = os.path.abspath(os.path.dirname(os.path.dirname('main.py')))
sys.path.append(WORKPATH)
from distbelief.utils.messaging import GradientServer
from distbelief.utils.serialization import ravel_model_params

from distbelief.utils import messaging, constant

import argparse
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from distbelief.optim import GradientSGD

from datetime import datetime
from example.models import *
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

import torch.optim as optim
from distbelief.server import GradientExecutor
import torch.multiprocessing as mp

net = ResNet18()
constant.MODEL_SIZE = ravel_model_params(net)


def get_dataset(args, transform):
    """
    :param args:
    :param transform:
    :return:
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
    global net

    logs = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainloader, testloader = get_dataset(args, transform)
    if args.cuda:
        net = net.cuda()

    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    else:
        print('distributed model')
        optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net)
        # optimizer = DownpourSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, min_lr=1e-5, cooldown=1,
                                                     factor=0.25)

    # train
    net.train()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        print("Training for epoch {}".format(epoch))
        net.train()
        # set distributed_sampler.epoch to shuffle data.
        trainloader.sampler.set_epoch(epoch)
        start = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
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
    global net
    if args.cuda:
        net = net.cuda()
    # if messaging.isCUDA:
    #     model.cuda()
    # gradient_warehouse = GradientWarehouse(worker_num=args.world_size, model=model)
    threads_num = dist.get_world_size() - 1
    # mp.set_start_method('spawn')
    size_list = [i.data.numel() for i in net.parameters()]
    threads = []
    procs = []
    global_model = ravel_model_params(net, cuda=True)
    global_model.share_memory_()
    synced_model = global_model.clone()
    synced_model.share_memory_()
    # shared_tensors = [synced_model.clone() for _ in range(args.world_size - 1)]
    shared_list = Manager().list([0 for _ in range(args.world_size - 1)])

    # print(shared_list)
    for i in range(1, threads_num + 1):
        # listener = GradientMessageListener(model_size=ravel_model_params(model).numel(), source=i)
        share_tensor = synced_model.clone()
        share_tensor.share_memory_()
        share_queue_recv = mp.Queue()
        share_queue_send = mp.Queue()
        th = GradientServer(share_tensor, share_queue_recv, share_queue_send,
                            model_size=ravel_model_params(net).numel(), source=i)
        th.start()
        p = GradientExecutor(share_tensor, share_queue_recv, share_queue_send, shared_list, rank=i,
                             worker_num=args.world_size,
                             global_model=global_model,
                             synced_model=synced_model, size_list=size_list)
        p.start()
        threads.append(th)
        procs.append(p)
    for t in threads:
        t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
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
    if args.cuda and not args.server:
        if socket.gethostname() == 'yan-pc':
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 1)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % (args.rank % 2)
        print('Using device%s, device count:%d' % (os.environ['CUDA_VISIBLE_DEVICES'], torch.cuda.device_count()))
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
        if args.server:
            import glob

            for infile in glob.glob(os.path.join(WORKPATH, '*.size')):
                os.remove(infile)
        if args.server:
            mp.set_start_method('spawn', force=True)

        dist.init_process_group('tcp', init_method='file://%s/sharedfile' % WORKPATH, group_name='mygroup',
                                world_size=args.world_size, rank=args.rank)
        if args.cuda:
            messaging.isCUDA = 1
        if args.server:
            # mp.set_start_method('spawn')
            init_server(args)
    main(args)
