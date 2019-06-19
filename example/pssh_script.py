# from __future__ import print_function
import argparse
import threading

from pssh.clients import ParallelSSHClient
from pssh.utils import enable_host_logger

enable_host_logger()

stdout = []


def Print(host, host_out):
    try:
        for line in host_out.stdout:
            stdout.append(line)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distbelief training example')
    parser.add_argument('--where', type=str, default='522', help='522 or th')
    args = parser.parse_args()
    # where = '522'
    print(args)
    if args.where == '522':
        hosts = ['192.168.3.100', '192.168.3.101', '192.168.3.102', '192.168.3.103', '192.168.3.104']
        # hosts = ['192.168.3.100', '192.168.3.101', '192.168.3.101', '192.168.3.102', '192.168.3.102', '192.168.3.103',
        #          '192.168.3.103', '192.168.3.104', '192.168.3.104']
        client = ParallelSSHClient(hosts, user='yan',
                                   proxy_host='172.18.233.36', proxy_user='yan',
                                   proxy_port=10000)
        host_args = ['--rank %d' % i for i in range(len(hosts))]
        # command = '/home/yan/anaconda3/bin/python /share/distbelief/example/main.py --dataset cifar10 --mode gradient_sgd --lr 0.1 --world-size ' + str(
        #     len(hosts)) + ' --cuda %s'
        # command = '/home/yan/anaconda3/bin/python /share/distbelief/example/main.py --mode gradient_sgd --world-size ' + str(len(hosts)) + ' --cuda %s'
        # command = '/home/yan/anaconda3/envs/an4/bin/python /share/distbelief/example/main.py  --dataset an4 --mode gradient_sgd --world-size ' + str(len(hosts)) + ' --cuda %s'
        command = '/home/yan/anaconda3/envs/an4/bin/python /share/distbelief/deepspeech/train.py --cuda --epoch 100 --lr 4e-4 --weight-decay 2e-5 --learning-anneal 1.01 --momentum 0.9 --num-workers 4 --augment --batch-size 5 --world-size ' + str(
            len(hosts)) + ' --cuda %s'
    else:
        hosts = ['gn16', 'gn17', 'gn17', 'gn18', 'gn18']
        client = ParallelSSHClient(hosts, )
        host_args = ['--rank %d' % i for i in range(len(hosts))]
        command = '~/anaconda3/bin/python /HOME/sysu_wgwu_2/WORKSPACE/GradientServer/distbelief/example/main.py --mode asgd --world-size ' + str(
            len(hosts)) + ' --cuda %s'

    output = client.run_command(command, host_args=host_args, use_pty=True, timeout=1000)
    threads = []
    for host, host_out in output.items():
        t = threading.Thread(target=Print, args=(host, host_out))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()
    for host, host_out in output.items():
        client.host_clients[host].close_channel(host_out.channel)
    # Join is not strictly needed here as channel has already been closed and
    # command has finished, but is safe to use regardless.
    client.join(output)
    # print(output)
