# from __future__ import print_function
import argparse
import threading

from pssh.clients import ParallelSSHClient
from pssh.utils import enable_host_logger

enable_host_logger()

stdout = []


def Print(host, host_out):
    for line in host_out.stdout:
        try:
            stdout.append(line)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--where', type=str, default='522', help='522 or th')
    args = parser.parse_args()
    # where = '522'
    print(args)
    if args.where == '522':
        threads = []
        server = ['192.168.3.100']
        worker = ['192.168.3.101', '192.168.3.102', '192.168.3.103', '192.168.3.104']
        process_per_worker = 1
        # command = '/home/yan/anaconda3/bin/python /share/DGS/example/main.py --dataset cifar10 --batch-size 64 --mode gradient_sgd --lr 0.1 --world-size ' + str(
        #     len(hosts)) + ' --cuda %s'
        # command = '/home/yan/anaconda3/bin/python /share/DGS/example/main.py --mode gradient_sgd --world-size ' + str(len(hosts)) + ' --cuda %s'
        # command = '/home/yan/anaconda3/envs/an4/bin/python /share/DGS/example/main.py  --dataset an4 --mode gradient_sgd --world-size ' + str(len(hosts)) + ' --cuda %s'
        # command = '/home/yan/anaconda3/envs/an4/bin/python /share/DGS/deepspeech/train.py --cuda --learning-anneal 1.01 --momentum 0.9 --num-workers 4 --augment --batch-size 5 --world-size ' + str(
        #     len(hosts)) + ' --cuda %s'

        # server
        host_args = ['--rank %d' % 0]
        client = ParallelSSHClient(server, timeout=10000, proxy_host='172.18.233.41', proxy_user='yan',
                                   proxy_port=10000, )
        command = '/home/yan/anaconda3/envs/torch1.3/bin/python /share/DGS/example/Imagenet_dist.py --world-size ' + str(
            len(worker) * process_per_worker + 1) + ' %s'
        output = client.run_command(command, host_args=host_args, use_pty=True, timeout=10000)
        for host, host_out in output.items():
            t = threading.Thread(target=Print, args=(host, host_out))
            t.start()
            threads.append(t)
        # worker
        for i in range(process_per_worker):
            host_args = ['--rank %d' % (j * process_per_worker - i) for j in range(1, len(worker) + 1)]
            print(host_args)
            client = ParallelSSHClient(worker, timeout=10000, proxy_host='172.18.233.41', proxy_user='yan',
                                       proxy_port=10000, )
            command = '/home/yan/anaconda3/envs/torch1.3/bin/python /share/DGS/example/Imagenet_dist.py --world-size ' + str(
                len(worker) * process_per_worker + 1) + ' %s'
            output = client.run_command(command, host_args=host_args, use_pty=True, timeout=10000)
            for host, host_out in output.items():
                t = threading.Thread(target=Print, args=(host, host_out))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()
        exit(12580)
    elif args.where == 'th':
        pass
        # hosts = ['gn22', 'gn17', 'gn17', 'gn18', 'gn18']
        # hosts = ['gpu30']
        # for i in ['30']:
        #     hosts.append('gpu%s' % str(i))
        #     hosts.append('gpu%s' % str(i))
        #     hosts.append('gpu%s' % str(i))
        #     hosts.append('gpu%s' % str(i))
        # # hosts = hosts[1:]
        # print('hosts:', hosts)
        # client = ParallelSSHClient(hosts, timeout=1000)
        # host_args = ['--rank %d' % i for i in range(len(hosts))]
        # # command = '~/anaconda3/bin/python /WORK/sysu_wgwu_2/GradientServer/DGS/example/main.py --dataset cifar10 --batch-size 16 --mode gradient_sgd --lr 0.1 --world-size ' + str(
        # #     len(hosts)) + ' --cuda %s'
        # command = '/GPUFS/app_GPU/application/anaconda3/5.3.1/envs/pytorch-py36/bin/python /GPUFS/sysu_wgwu_8/GradientServer/DGS/example/Imagenet_dist.py -data /GPUFS/sysu_wgwu_8/ImageNet --world-size ' + str(
        #     len(hosts)) + ' %s'
    elif args.where == 'v100':
        threads = []
        server = ['gpu2']
        worker = ['gpu23', 'gpu29', 'gpu49', 'gpu55']
        process_per_worker = 4
        # server
        host_args = ['--rank %d' % 0]
        client = ParallelSSHClient(server, timeout=10000)
        command = '/GPUFS/app_GPU/application/anaconda3/5.3.1/envs/pytorch-py36/bin/python /GPUFS/sysu_wgwu_8/GradientServer/DGS/example/Imagenet_dist.py -data /GPUFS/sysu_wgwu_8/ImageNet --world-size ' + str(
            len(worker) * process_per_worker + 1) + ' %s'
        output = client.run_command(command, host_args=host_args, use_pty=True, timeout=10000)
        for host, host_out in output.items():
            t = threading.Thread(target=Print, args=(host, host_out))
            t.start()
            threads.append(t)
        # worker
        for i in range(1, process_per_worker + 1):
            host_args = ['--rank %d' % j for j in range(i, i + len(worker) * process_per_worker, process_per_worker)]
            print(host_args)
            client = ParallelSSHClient(worker, timeout=10000)
            command = '/GPUFS/app_GPU/application/anaconda3/5.3.1/envs/pytorch-py36/bin/python /GPUFS/sysu_wgwu_8/GradientServer/DGS/example/Imagenet_dist.py -data /GPUFS/sysu_wgwu_8/ImageNet --epoch 120 --momentum 0.45 --world-size ' + str(
                len(worker) * process_per_worker + 1) + ' %s'
            output = client.run_command(command, host_args=host_args, use_pty=True, timeout=10000)
            for host, host_out in output.items():
                t = threading.Thread(target=Print, args=(host, host_out))
                t.start()
                threads.append(t)
        for thread in threads:
            thread.join()
        exit(12580)
