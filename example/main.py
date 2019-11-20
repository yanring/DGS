import os
import sys

WORKPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(WORKPATH)
sys.path.append(WORKPATH)

from core.utils.serialization import ravel_model_params

from core.utils import constant
import torch.distributed as dist
from core.server import GradientServer


def init_server(args, net):
    print('init server!!!')
    dist.init_process_group('gloo', init_method='file://%s/sharedfile' % WORKPATH, group_name='mygroup',
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
