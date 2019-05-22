# 
"""
Parameter server for distbelief
"""

import logging
import threading
import torch
import torch.optim

from distbelief.utils.messaging import MessageCode, MessageListener, send_message, GSMessageCode, \
    GradientMessageListener
from distbelief.utils.serialization import ravel_model_params, unravel_sparse_gradient, ravel_sparse_gradient, \
    server_gradient_filter

_LOGGER = logging.getLogger(__name__)
cond = threading.Condition()


class ParameterServer(MessageListener):
    """ParameterServer"""

    def __init__(self, model):
        _LOGGER.info("Creating ParameterServer")
        print("Creating ParameterServer")
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        # init superclass
        super(ParameterServer, self).__init__(model)

    def receive(self, sender, message_code, parameter):
        print("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            # be sure to clone here
            self.parameter_shard = parameter.clone()

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)

        elif message_code == MessageCode.GradientUpdate:
            self.parameter_shard.add_(parameter)


#
# class GradientWarehouse2:
#     """Warehouse for gradient, store multiple version of gradient"""
#
#     def __init__(self, version_num=10, worker_num=3, model=None):
#         self.gradient_storage = {}
#         self.gradient_storage_state = {}

un_synced_worker = set()


class GradientServer(GradientMessageListener):
    """GradientServer"""

    def __init__(self, model, rank=0, worker_num=None, global_model=None, synced_model=None, momentum=None):
        _LOGGER.info("Creating GradientServer")
        print("Creating GradientServer")
        # self.gradient_warehouse = gradient_warehouse
        # self.net = model
        self.max_version = 0
        self.worker_count = 0
        self.worker_num = worker_num
        self.global_model = global_model
        self.global_model.share_memory_()
        super(GradientServer, self).__init__(model_size=global_model.numel(), source=rank)
        self.synced_model = synced_model
        self.synced_model.share_memory_()
        self.synced_version = 0
        self.acc_send_grad = synced_model.clone().zero_()
        self.acc_send_grad.share_memory_()
        self.momentum = momentum
        # self.pool = mp.Pool(processes=1)
        self.net = model
        if rank == 1:
            for i in range(1, self.worker_num):
                self.sync_worker_model(i, 1)
        self.node_gradient = {}

    def sync_worker_model(self, sender, version):
        # if sender == 2:
        model = self.synced_model
        send_message(GSMessageCode.ModelUpdate, model, dst=sender, gradient_version=version)

    def sync_model(self):
        self.synced_model.copy_(self.global_model)
        # self.synced_version = self
        return self.synced_model

    def update(self, rank, version, gradient_update):
        """
        :param momentum:
        :param rank: rank of worker node
        :param version: version of gradient
        :param gradient_update: tensor, gradient update tensor
        :return:
        """
        print("update gradient from rank%d,version%d" % (rank, version))
        # self.momentum.mul_(0.5)
        old_model = self.global_model.clone()
        self.global_model.add_(-1, gradient_update)
        self.global_model.add_(self.momentum.mul_(0.5))
        self.momentum = self.global_model - old_model

        # agg_gradient = self.global_model.add(-1, self.synced_model)

        return None, version

    def receive(self, sender, message_code, gradient_version, parameter):
        global un_synced_worker
        print("rank {} Processing message: {} from sender {} gradient version {}".format(self.source, message_code.name,
                                                                                         sender,
                                                                                         gradient_version))
        self.max_version = max(self.max_version, gradient_version)

        if message_code == GSMessageCode.GradientUpdate:
            self.update(sender, gradient_version, parameter)
            send_message(GSMessageCode.ModelUpdate, self.global_model, dst=sender,
                         gradient_version=gradient_version)
        elif message_code == GSMessageCode.SparseGradientUpdate:
            # parameter = unravel_sparse_gradient(parameter)
            send_grad = update(self.global_model, self.synced_model, self.acc_send_grad, parameter)

            # send_grad = self.pool.apply(update, (self.global_model, self.synced_model, self.acc_send_grad, parameter))
            # agg_gradient, new_version = self.update(sender, gradient_version, parameter)
            if sender == 1 and self.max_version % 100 is 1 and gradient_version > 20:
                self.sync_model()
                un_synced_worker = set(range(1, self.worker_num))
            if sender in un_synced_worker:
                self.acc_send_grad.zero_()
                self.sync_worker_model(sender, gradient_version)
                un_synced_worker.remove(sender)
            else:
                # send_grad = agg_gradient.add(-1, self.acc_send_grad)
                # server_gradient_filter(send_grad, rate=0.01)
                # self.acc_send_grad.add_(send_grad)
                # send_grad = ravel_sparse_gradient(send_grad)
                send_message(GSMessageCode.SparseGradientUpdate, send_grad,
                             dst=sender,
                             gradient_version=gradient_version)

        else:
            raise Exception('GSMessageCode not implemented')


def update(global_model, synced_model, acc_send_grad, parameter):
    parameter = unravel_sparse_gradient(parameter)
    global_model.add_(-1, parameter)
    agg_gradient = global_model.add(-1, synced_model)
    send_grad = agg_gradient.add(-1, acc_send_grad)
    server_gradient_filter(send_grad, rate=0.01)
    acc_send_grad.add_(send_grad)
    send_grad = ravel_sparse_gradient(send_grad)
    return send_grad

# def process_sparse_gradient
