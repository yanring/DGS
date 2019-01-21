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


class GradientWarehouse:
    """Warehouse for gradient, store multiple version of gradient"""

    def __init__(self, version_num=10, worker_num=3, model=None):
        self.gradient_storage = {}
        self.gradient_storage_state = {}
        self.max_version = 0
        self.worker_count = 0
        self.worker_num = worker_num
        self.global_model = ravel_model_params(model)
        self.synced_model = ravel_model_params(model)
        self.synced_version = 0

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

        self.max_version = max(self.max_version, version)

        return agg_gradient, version

    def sync_model(self):
        self.synced_model = self.global_model.clone()
        # self.synced_version = self
        return self.synced_model


class GradientServer(GradientMessageListener):
    """GradientServer"""

    def __init__(self, model, gradient_warehouse, storage_num=10, rank=0):
        _LOGGER.info("Creating GradientServer")
        print("Creating GradientServer")
        # self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.gradient_warehouse = gradient_warehouse
        self.rank = rank
        super(GradientServer, self).__init__(model)
        self.model = torch.zeros(ravel_model_params(model).numel())
        self.gradient_warehouse.model = self.model
        self.sync_worker_model(1, 1)
        self.sync_worker_model(2, 1)
        self.un_synced_worker = set()

    def sync_worker_model(self, sender, version):
        # if sender == 2:
        model = self.gradient_warehouse.synced_model
        send_message(GSMessageCode.ModelUpdate, model, dst=sender, gradient_version=version)

    def receive(self, sender, message_code, gradient_version, parameter):
        print("rank {} Processing message: {} from sender {} gradient version {}".format(self.rank, message_code.name,
                                                                                         sender,
                                                                                         gradient_version))

        if message_code == GSMessageCode.GradientUpdate:
            agg_gradient, new_version = self.gradient_warehouse.update(sender, gradient_version, parameter)

            if sender == 1 and self.gradient_warehouse.max_version % 100 is 1:
                self.gradient_warehouse.sync_model()
                self.un_synced_worker = set(range(1, self.gradient_warehouse.worker_num))
            if sender in self.un_synced_worker:
                self.sync_worker_model(sender, new_version)
                self.un_synced_worker.remove(sender)
            else:
                # print('synced_model:', self.gradient_warehouse.synced_model)
                # print('updated_model:', self.gradient_warehouse.synced_model.add(agg_gradient))
                send_message(GSMessageCode.GradientUpdate, agg_gradient,
                             dst=sender,
                             gradient_version=new_version)
        else:
            raise Exception('GSMessageCode not implemented')
