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

    def __init__(self, version_num=10, worker_num=3):
        self.gradient_storage = {}
        self.gradient_storage_state = {}
        self.version_num = version_num
        self.worker_count = 0
        self.worker_num = worker_num

    def update(self, rank, version, gradient_update):
        """
        :param rank: rank of worker node
        :param version: version of gradient
        :param gradient_update: tensor, gradient update tensor
        :return:
        """
        print("update gradient from rank%d,version%d" % (rank, version))
        if version in self.gradient_storage.keys():
            # version exist
            # TODO: use add_ or average?
            self.gradient_storage[version].add_(gradient_update)
            self.gradient_storage_state[version].add(rank)
        else:
            # version does not exist
            if len(self.gradient_storage) > self.version_num:

                # pop the last one
                lowest_gradient_key = min(self.gradient_storage_state)

                if version < lowest_gradient_key:
                    # find a drop out node, sync node
                    print("gradient version lower than lowest_gradient_key")
                    # self.gradient_storage[version] = gradient_update.clone()
                    # self.gradient_storage_state[version] = {rank}
                    # return gradient_update, version
                    # raise Exception('find a drop out node which should be synced!')
                else:
                    self.gradient_storage.pop(lowest_gradient_key)
                    self.gradient_storage_state.pop(lowest_gradient_key)
                    print("pop version:%d gradient" % lowest_gradient_key)
                    # add new one
                    self.gradient_storage[version] = gradient_update.clone()
                    self.gradient_storage_state[version] = {rank}
            else:
                self.gradient_storage[version] = gradient_update.clone()
                self.gradient_storage_state[version] = {rank}

        bound_version = max(self.gradient_storage) - self.version_num
        print('bound_version=%d' % bound_version)
        if bound_version >= version > self.version_num:
            # TODO: check node state, sync nodes which slower than self.version_num version
            # find a drop out node, sync this node
            sync_to = max(self.gradient_storage) - 2
            # agg gradient from version to sync_to
            gradient_update.add_(self.get_bunch([i for i in range(version + 1, sync_to)]))
            print("sync node %d from version %d to version %d" % (rank, version, sync_to))
            return gradient_update, sync_to

        return self.gradient_storage[version], version

    def get(self, version):
        """
        get gradient
        :param version: gradient version
        :return:
        """
        try:
            return self.gradient_storage[version]

        except Exception as e:
            print("version:%d gradient does not exist" % version)
            raise e

    def get_bunch(self, version_list):
        """
        get sum of [version_list] gradients
        :param version_list:
        :return:
        """
        res = None
        for i in version_list:
            tmp_tensor = self.gradient_storage.get(i)
            if tmp_tensor is None:
                continue
            if res is None:
                res = tmp_tensor.clone()
            res.add_(tmp_tensor)
        return res

    def sync_model(self, rank, model):
        print('model from rank %d' % rank)
        self.model.add_(1 / (self.worker_num - 1), model)
        self.worker_count += 1
        with cond:
            if self.worker_count < self.worker_num - 1:
                cond.wait()
            else:
                print(self.worker_count)
                cond.notify()
            # print(self.worker_count)
        version = max(self.gradient_storage)
        return self.model, version


class GradientServer(GradientMessageListener):
    # TODO
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

    def receive(self, sender, message_code, gradient_version, parameter):
        print("rank {} Processing message: {} from sender {} gradient version {}".format(self.rank, message_code.name,
                                                                                         sender,
                                                                                         gradient_version))
        #
        # if message_code == GSMessageCode.ParameterUpdate:
        #     # be sure to clone here
        #     raise Exception("GSMessageCode.ParameterUpdate no implement")
        # self.parameter_shard = parameter.clone()

        if sender is 1 and gradient_version % 20 is 1 and message_code == GSMessageCode.GradientUpdate and gradient_version > 800:

            # if not self.gradient_warehouse.lock.locked():
            #     self.gradient_warehouse.lock.acquire()
            for i in range(1, self.gradient_warehouse.worker_num):
                print('Send model request to worker %d' % i)
                send_message(GSMessageCode.ModelRequest, self.model, dst=i,
                             gradient_version=0)
                # self.gradient_warehouse.lock.release()

        if message_code == GSMessageCode.ModelUpdate:
            model, new_version = self.gradient_warehouse.sync_model(sender, parameter)
            send_message(GSMessageCode.ModelUpdate, model, dst=sender,
                         gradient_version=new_version)
            print('Send updated model to worker %d' % sender)
            self.gradient_warehouse.worker_count -= 1
            if self.gradient_warehouse.worker_count == 0:
                self.gradient_warehouse.model = torch.zeros(self.gradient_warehouse.model.numel())

        elif message_code == GSMessageCode.GradientUpdate:
            print("update gradient_warehouse")
            agg_gradient, new_version = self.gradient_warehouse.update(sender, gradient_version, parameter)
            print("send aggregated gradient back")
            send_message(GSMessageCode.GradientUpdate, agg_gradient, dst=sender,
                         gradient_version=new_version)

        # if self.rank % 2:
        #     time.sleep(5)
