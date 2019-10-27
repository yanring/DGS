import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    losses = list()
    top1 = list()
    top5 = list()

    for i, (input, target) in enumerate(train_loader):

        # compute output
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)

        losses.append(loss.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.append(acc1[0].item())
        top5.append(acc5[0].item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            print(str(datetime.now()) + ' Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f} ({loss_avg:.4f})\t'
                  'Acc@1 {top1:.3f} ({top1_avg:.3f})\t'
                  'Acc@5 {top5:.3f} ({top5_avg:.3f})'.format(
                epoch, i, len(train_loader), loss=loss.item(), loss_avg=sum(losses) / float(len(losses)),
                top1=acc1[0].item(), top1_avg=sum(top1) / float(len(top1)),
                top5=acc5[0].item(), top5_avg=sum(top5) / float(len(top5))))


def validate(test_loader, model, criterion):
    # switch to train mode
    model.eval()
    losses = list()
    top1 = list()
    top5 = list()

    for i, (input, target) in enumerate(test_loader):

        # compute output
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)

        losses.append(loss.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.append(acc1[0].item())
        top5.append(acc5[0].item())

        if i % 10 == 0:
            print('[{0}/{1}]\t'
                  'Loss {loss:.4f} ({loss_avg:.4f})\t'
                  'Acc@1 {top1:.3f} ({top1_avg:.3f})\t'
                  'Acc@5 {top5:.3f} ({top5_avg:.3f})'.format(
                i, len(test_loader), loss=loss.item(), loss_avg=sum(losses) / float(len(losses)),
                top1=acc1[0].item(), top1_avg=sum(top1) / float(len(top1)),
                top5=acc5[0].item(), top5_avg=sum(top5) / float(len(top5))))

    print(' * Acc@1 {top1_avg:.3f} Acc@5 {top5_avg:.3f}'
          .format(top1_avg=sum(top1) / float(len(top1)), top5_avg=sum(top5) / float(len(top5))))


def main():
    args = parser.parse_args()
    args.gpu = 0
    # args.data = '/share/distbelief/data/tiny-imagenet-200/'
    args.data = '/home/yan/data/'
    traindir = os.path.join(args.data, 'train')  # /train/ を指定されたパスに追加
    testdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # 正規化定数

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),  # 画像をサイズ224に切り出しもしくはリサイズ
            transforms.RandomHorizontalFlip(),  # ランダムに画像をフリップ（水増し）
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)

    model = models.resnet18(num_classes=1000)
    # model = EfficientNet.from_name('efficientnet-b0')
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=1e-4)

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    for epoch in range(0, 100):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(test_loader, model, criterion)


main()
