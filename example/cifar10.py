import sys
import time

import os
from torch.optim.lr_scheduler import MultiStepLR

WORKPATH = os.path.abspath(os.path.dirname(os.path.dirname('main.py')))
sys.path.append(WORKPATH)

from distbelief.utils.serialization import ravel_model_params

from distbelief.utils import constant

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from distbelief.optim import GradientSGD

from datetime import datetime
from example.models import *
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

import torch.optim as optim


def get_dataset(args, transform_train, transform_test):
    """
    :param args:
    :param transform:
    :return:
    """
    if args.dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    sampler = DistributedSampler(trainset, args.world_size - 1, args.rank - 1)
    # sampler = DistributedSampler(trainset, 1, 0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                              sampler=sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    return trainloader, testloader


def cifar10(args):
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
    if args.cuda:
        net = net.cuda()
    constant.MODEL_SIZE = ravel_model_params(net).numel()
    if args.no_distributed:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    else:
        print('distributed model')
        optimizer = GradientSGD(net.parameters(), lr=args.lr, model=net, momentum=args.momentum, weight_decay=5e-4,
                                args=args)
        # optimizer = DownpourSGD(net.parameters(), lr=args.lr, n_push=args.num_push, n_pull=args.num_pull, model=net)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, cooldown=1, verbose=True, factor=0.25)
    scheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)
    compress_ratio = [0.001] * args.epochs
    compress_ratio[0:4] = [0.1, 0.0625, 0.0625 * 0.25, 0.004]
    # train
    net.train()

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # scheduler.step()
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
        if args.no_distributed or args.rank == 1:
            # scheduler.step(logs[-1]['test_loss'])
            scheduler.step()

    df = pd.DataFrame(logs)
    print(df)
    if args.no_distributed:
        if args.cuda:
            df.to_csv('log/gpu.csv', index_label='index')
        else:
            df.to_csv('log/single.csv', index_label='index')
    else:
        df.to_csv('log/node{}_{}_{}_m{}_{}worker.csv'.format(args.rank - 1, args.mode,
                                                             args.model, args.momentum, args.world_size - 1),
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
