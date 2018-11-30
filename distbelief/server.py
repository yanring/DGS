# 
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
from distbelief.utils.messaging import MessageCode, MessageListener, send_message, GSMessageCode, \
    GradientMessageListener
from distbelief.utils.serialization import ravel_model_params, unravel_model_params

_LOGGER = logging.getLogger(__name__)


class ParameterServer(MessageListener):
    """ParameterServer"""

    def __init__(self, model):
        _LOGGER.info("Creating ParameterServer")
        print("Creating ParameterServer")
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        # init superclass
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        print("Processing message: {} from sender {}".format(message_code.name, sender))

        if message_code == MessageCode.ParameterUpdate:
            # be sure to clone here
            # 因为只是个引用，所以必须要克隆 yan
            self.parameter_shard = parameter.clone()

        elif message_code == MessageCode.ParameterRequest:
            send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)

        elif message_code == MessageCode.GradientUpdate:
            self.parameter_shard.add_(parameter)


class GradientWarehouse:
    """Warehouse for gradient, store multiple version of gradient"""

    def __init__(self, model, version_num=10):
        self.gradient_storage = {}
        self.gradient_storage_state = {}
        self.version_num = version_num

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
                    # print("gradient version lower than lowest_gradient_key")
                    # self.gradient_storage[version] = gradient_update.clone()
                    # self.gradient_storage_state[version] = {rank}
                    # return gradient_update, version
                    raise Exception('find a drop out node which should be synced!')

                self.gradient_storage.pop(lowest_gradient_key)
                self.gradient_storage_state.pop(lowest_gradient_key)
                print("pop version:%d gradient" % lowest_gradient_key)

                # add new one
                self.gradient_storage[version] = gradient_update.clone()
                self.gradient_storage_state[version] = {rank}
            else:
                self.gradient_storage[version] = gradient_update.clone()
                self.gradient_storage_state[version] = {rank}

        bound_version = max(self.gradient_storage_state) - self.version_num
        if bound_version >= version > self.version_num + 10:
            # TODO: check node state, sync nodes which slower than self.version_num version
            # find a drop out node, sync this node
            sync_to = max(self.gradient_storage_state) - 2
            gradient_update = self.gradient_storage[bound_version].clone()
            gradient_update.add_(self.get_bunch([i for i in range(version, sync_to)]))
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
        res = self.gradient_storage[version_list[0]].clone()
        for i in version_list[1:]:
            res.add_(self.gradient_storage[i])
        return res


class GradientServer(GradientMessageListener):
    # TODO
    """GradientServer"""

    def __init__(self, model, storage_num=10):
        _LOGGER.info("Creating GradientServer")
        print("Creating GradientServer")
        # self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        self.gradient_warehouse = GradientWarehouse(model, storage_num)
        super().__init__(model)

    def receive(self, sender, message_code, gradient_version, parameter):
        print("Processing message: {} from sender {} gradient version {}".format(message_code.name, sender,
                                                                                 gradient_version))
        #
        # if message_code == GSMessageCode.ParameterUpdate:
        #     # be sure to clone here
        #     # 因为只是个引用，所以必须要克隆 yan
        #     raise Exception("GSMessageCode.ParameterUpdate no implement")
        # self.parameter_shard = parameter.clone()

        if message_code == GSMessageCode.GradientRequest:
            send_message(GSMessageCode.ParameterUpdate, self.gradient_warehouse.get(gradient_version), dst=sender,
                         gradient_version=gradient_version)

        elif message_code == GSMessageCode.GradientUpdate:
            print("update gradient_warehouse")
            agg_gradient, new_version = self.gradient_warehouse.update(sender, gradient_version, parameter)
            print("send aggregated gradient back")
            send_message(GSMessageCode.GradientUpdate, agg_gradient, dst=sender,
                         gradient_version=new_version)
