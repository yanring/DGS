# 
"""
Parameter server for distbelief
"""

import logging
import torch
import torch.optim

from distbelief.utils.messaging import MessageCode, MessageListener, send_message, GSMessageCode, \
    GradientMessageListener
from distbelief.utils.serialization import ravel_model_params

_LOGGER = logging.getLogger(__name__)


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

    def __init__(self, version_num=10, worker_num=2):
        self.worker_num = worker_num
        self.gradient_storage = {}
        self.gradient_storage_state = {}
        self.version_num = version_num
        self.triggers = {}  # Triggers are set by fast nodes who need slow nodes' gradients.
        for i in range(1, worker_num + 1):
            self.triggers[i] = []

    def update(self, rank, version, gradient_update):
        """
        :param rank: rank of worker node
        :param version: version of gradient
        :param gradient_update: tensor, gradient update tensor
        :return:
        """
        print("update gradient from rank%d,version%d" % (rank, version))
        fast_flag = 0
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

        # sync slow nodes
        bound_version = max(self.gradient_storage) - self.version_num
        print('bound_version=%d' % bound_version)
        if bound_version >= version > self.version_num:
            # TODO: check node state, sync nodes which slower than self.version_num version
            # find a drop out node, sync this node
            sync_to = max(self.gradient_storage) - 2
            # agg gradient from version to sync_to
            gradient_update.add_(self.get_bunch([i for i in range(version + 1, sync_to)]))
            print("sync node %d from version %d to version %d" % (rank, version, sync_to))
            return gradient_update, sync_to, [0], fast_flag

        # sync fast nodes
        if len(self.gradient_storage_state[version]) <= self.worker_num / 2:
            # set trigger and fast flag
            self.triggers[rank].append(version)
            fast_flag = 1

        sync_fast_flag = 0
        trigger_expired_list = []
        trigger_used_list = []
        for trigger in self.triggers[rank]:
            # check old triggers
            if trigger <= bound_version:
                # gradient expired
                trigger_expired_list.append(trigger)
                continue
            if len(self.gradient_storage_state[trigger]) > self.worker_num / 2:
                if not sync_fast_flag:
                    gradient_update = self.gradient_storage[version].clone()
                    sync_fast_flag = 1
                gradient_update.add_(self.gradient_storage[trigger])
                trigger_used_list.append(trigger)
                print("Sync-fast: change fast node rank %d gradient version %d" % (rank, trigger))
                break

        for trigger in trigger_used_list + trigger_expired_list:
            # remove used and expired triggers
            self.triggers[rank].remove(trigger)

        if sync_fast_flag:
            # trigger triggered
            return gradient_update, version, trigger_used_list, fast_flag

        return self.gradient_storage[version], version, [0], fast_flag

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


class GradientServer(GradientMessageListener):
    # TODO
    """GradientServer"""

    def __init__(self, model, gradient_warehouse, storage_num=10, rank=0):
        _LOGGER.info("Creating GradientServer")
        print("Creating GradientServer")
        # self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        self.gradient_warehouse = gradient_warehouse
        self.rank = rank
        super(GradientServer, self).__init__(model)

    def receive(self, sender, message_code, gradient_version, trigger, fast_flag, parameter):
        print("rank {} Processing message: {} from sender {} gradient version {}".format(self.rank, message_code.name,
                                                                                         sender,
                                                                                         gradient_version))
        #
        # if message_code == GSMessageCode.ParameterUpdate:
        #     # be sure to clone here
        #     raise Exception("GSMessageCode.ParameterUpdate no implement")
        # self.parameter_shard = parameter.clone()

        if message_code == GSMessageCode.GradientRequest:
            send_message(GSMessageCode.ParameterUpdate, self.gradient_warehouse.get(gradient_version), dst=sender,
                         gradient_version=gradient_version)

        elif message_code == GSMessageCode.GradientUpdate:
            # print("update gradient_warehouse")
            agg_gradient, new_version, triggers, fast_flag = self.gradient_warehouse.update(sender, gradient_version,
                                                                                            parameter)
            # print("send aggregated gradient back")
            send_message(GSMessageCode.GradientUpdate, agg_gradient, dst=sender,
                         gradient_version=new_version, trigger=triggers[0], fast_flag=fast_flag)

        # if self.rank % 2:
        #     time.sleep(5)
