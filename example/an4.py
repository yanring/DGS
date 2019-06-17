import json
import os
import sys
import time

import torch.optim as optim
import torch.utils.data.distributed
from torch_baidu_ctc import CTCLoss
from tqdm import tqdm

WORKPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKPATH)
sys.path.append(WORKPATH + '/deepspeech')

from data.data_loader import AudioDataLoader, SpectrogramDataset, DistributedBucketingSampler
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns
from datetime import datetime
import pandas as pd

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
device = torch.device("cuda")


def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


Total_param_num = 0
Sparse_param_num = 0
criterion = CTCLoss()
best_wer = None
decoder = None
audio_conf = None
labels = None


# 获取数据


def init_net(args):
    # Model
    global decoder, audio_conf, labels
    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    rnn_type = args.rnn_type.lower()
    decoder = GreedyDecoder(labels)
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    net = DeepSpeech(rnn_hidden_size=args.hidden_size,
                     nb_layers=args.hidden_layers,
                     labels=labels,
                     rnn_type=supported_rnns[rnn_type],
                     audio_conf=audio_conf,
                     bidirectional=args.bidirectional)
    net = net.cuda()
    return net


def an4(args, optimizer, net):
    avg_loss, start_epoch, start_iter = 0, 0, 0

    # Training Data
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=True)
    train_sampler = DistributedBucketingSampler(train_dataset, batch_size=args.batch_size,
                                                num_replicas=args.world_size - 1, rank=args.rank - 1)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler)

    # Testing Data
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Optimizer and scheduler of Training
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=args.learning_anneal)

    logs = []
    print("Training Start")
    losses = AverageMeter()

    for epoch in range(args.epochs):

        print("Training for epoch {}".format(epoch))
        net.train()

        for i, (data) in enumerate(train_loader):
            batch_start_time = time.time()

            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            inputs = inputs.to(device)

            optimizer.zero_grad()

            out, output_sizes = net(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            # Loss Operation
            loss = criterion(out, targets, output_sizes, target_sizes).to(device)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            inf = float("inf")

            loss_value = loss.item()

            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            loss.backward()

            # Gradient Clip
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_norm)

            # paralist = gradient_execute(net)

            # SGD step
            optimizer.step()

            # for para1, para2 in zip(paralist, net.parameters()):
            #     para2.grad.data = para1

            log_obj = {
                'timestamp': datetime.now(),
                'iteration': i,
                'training_loss': losses.avg,
                'total_param': Total_param_num,
                'sparse_param': Sparse_param_num,
                'mini_batch_time': (time.time() - batch_start_time)
            }
            logs.append(log_obj)


            if i % 5 == 0:
                print("Timestamp: {timestamp} | "
                      "Iteration: {iteration:6} | "
                      "Loss: {training_loss:6.4f} | "
                      "Total_param: {total_param:6} | "
                      "Sparse_param: {sparse_param:6} | "
                      "Mini_Batch_Time: {mini_batch_time:6.4f} | ".format(**log_obj))


        # if True:
        test_wer, test_cer = evaluate(net, test_loader)
        logs[-1]['test_wer'], logs[-1]['test_cer'] = test_wer, test_cer

        print("Timestamp: {timestamp} | "
              "Iteration: {iteration:6} | "
              "Loss: {training_loss:6.4f} | "
              "Total_param: {total_param:6} | "
              "Sparse_param: {sparse_param:6} | "
              "Mini_Batch_Time: {mini_batch_time:6.4f} | "
              "Test Wer: {test_wer:6.4f} | "
              "Test Cer: {test_cer:6.4f}".format(**logs[-1]))

        # sche_wer, sche_cer = evaluate(net, test_loader)
        scheduler.step(test_wer)

    df = pd.DataFrame(logs)
    df.to_csv(WORKPATH + '/log/node{}_{}_{}_m{}_e{}_b{}_{}worker_{}.csv'.format(args.rank - 1, args.mode,
                                                                                'an4', args.momentum,
                                                                                args.epochs,
                                                                                args.batch_size,
                                                                                args.world_size - 1,
                                                                                test_wer))
    # df.to_csv('./log/{}_Node{}_{}.csv'.format(args.file_name, args.dist_rank, datetime.now().strftime("%Y-%m-%d %H:%M:%S")), index_label='index')
    print("Finished Training")


def evaluate(net, test_loader):
    total_cer, total_wer = 0, 0
    net.eval()
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            inputs = inputs.to(device)

            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size

            out, output_sizes = net(inputs, input_sizes)

            decoded_output, _ = decoder.decode(out, output_sizes)
            target_strings = decoder.convert_to_strings(split_targets)
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                transcript, reference = decoded_output[x][0], target_strings[x][0]
                wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                cer += decoder.cer(transcript, reference) / float(len(reference))
            total_cer += cer
            total_wer += wer
            del out
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
    print('Validation Summary Epoch: [{0}]\t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(-1, wer=wer, cer=cer))
    return wer, cer


if __name__ == "__main__":
    pass
    # dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.dist_rank,world_size=args.world_size)
    # dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size,group_name='mygroup')
    # an4(args2,None)
