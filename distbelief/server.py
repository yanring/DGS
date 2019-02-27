# 
"""
Parameter server for distbelief
"""

import logging
import threading
import torch
import torch.optim
from torch.multiprocessing import Process

from distbelief.utils.messaging import MessageCode, MessageListener, send_message, GSMessageCode
from distbelief.utils.serialization import ravel_model_params

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

# un_synced_worker = set()


class GradientExecutor(Process):
    """GradientExecutor"""

    def __init__(self, shared_gradient, shared_queue_recv, shared_queue_send, shared_list, rank=0, worker_num=None,
                 global_model=None,
                 synced_model=None):
        super().__init__()
        _LOGGER.info("Creating GradientExecutor")
        print("Creating GradientExecutor")
        self.max_version = 0
        self.worker_count = 0
        self.worker_num = worker_num
        self.global_model = global_model
        self.synced_model = synced_model
        self.synced_version = 0
        self.acc_send_grad = synced_model.clone().zero_()
        self.shared_gradient = shared_gradient
        self.shared_queue_recv = shared_queue_recv
        self.shared_queue_send = shared_queue_send
        self.shared_list = shared_list
        self.net = None
        self.rank = rank
        if rank == 1:
            for i in range(1, self.worker_num):
                self.sync_worker_model(i, 1)
        self.node_gradient = {}

    def sync_worker_model(self, sender, version):
        # if sender == 2:
        model = self.synced_model
        # self.shared_gradient.copy_(model)
        self.send_message(self.synced_model, GSMessageCode.ModelUpdate, version)
        # send_message(GSMessageCode.ModelUpdate, model, dst=sender, gradient_version=version)

    def sync_model(self):
        self.synced_model.copy_(self.global_model)
        # self.synced_version = self
        return self.synced_model

    def update(self, rank, version, gradient_update):
        """
        :param rank: rank of worker node
        :param version: version of gradient
        :param gradient_update: tensor, gradient update tensor
        :return:
        """
        print("update gradient from rank%d,version%d" % (rank, version))

        self.global_model.add_(-1, gradient_update)

        agg_gradient = self.global_model.add(-1, self.synced_model)

        return agg_gradient, version

    def receive(self, sender, message_code, gradient_version):
        # global un_synced_worker
        print("Processing message: {} from sender {} gradient version {}".format(message_code.name,
                                                                                 sender,
                                                                                 gradient_version))
        self.max_version = max(self.max_version, gradient_version)

        if message_code == GSMessageCode.GradientUpdate:
            self.update(sender, gradient_version, self.shared_gradient)
            send_message(GSMessageCode.ModelUpdate, self.global_model, dst=sender,
                         gradient_version=gradient_version)
        elif message_code == GSMessageCode.SparseGradientUpdate:
            # parameter = unravel_sparse_gradient(parameter)
            # send_grad = update(self.global_model, self.synced_model, self.acc_send_grad, self.shared_gradient)

            # send_grad = self.pool.apply(update, (self.global_model, self.synced_model, self.acc_send_grad, parameter))
            agg_gradient, new_version = self.update(sender, gradient_version, self.shared_gradient)
            if sender == 1 and self.max_version % 100 is 1 and gradient_version > 20:
                self.sync_model()
                for i in range(1, self.worker_num):
                    self.shared_list[i - 1] = 1
            if self.shared_list[self.rank - 1]:
                self.acc_send_grad.zero_()
                self.sync_worker_model(sender, gradient_version)
                self.shared_list[self.rank - 1] = 0
            else:
                send_grad = agg_gradient.add(-1, self.acc_send_grad)
                # server_gradient_filter(send_grad, rate=0.01)
                self.acc_send_grad.add_(send_grad)
                self.send_message(send_grad, GSMessageCode.SparseGradientUpdate, gradient_version)
                # send_grad = ravel_sparse_gradient(send_grad)
                # send_message(GSMessageCode.SparseGradientUpdate, send_grad,
                #              dst=sender,
                #              gradient_version=gradient_version)

        else:
            raise Exception('GSMessageCode not implemented')

    def send_message(self, payload, message_code, gradient_version):
        self.shared_gradient.copy_(payload)
        self.shared_queue_send.put([message_code, gradient_version])

    def run(self):
        while 1:
            recv = self.shared_queue_recv.get()
            # print(recv)
            self.receive(recv[0], recv[1], recv[2])
            print('Process %d is running' % self.rank)

# def update(global_model, synced_model, acc_send_grad, parameter):
#     parameter = unravel_sparse_gradient(parameter)
#     global_model.add_(-1, parameter)
#     agg_gradient = global_model.add(-1, synced_model)
#     send_grad = agg_gradient.add(-1, acc_send_grad)
#     server_gradient_filter(send_grad, rate=0.01)
#     acc_send_grad.add_(send_grad)
#     send_grad = ravel_sparse_gradient(send_grad)
#     return send_grad

# def process_sparse_gradient
