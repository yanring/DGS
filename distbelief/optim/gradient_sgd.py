import time

import logging
import torch
import torch.distributed as dist
from queue import Queue
from torch.optim.optimizer import Optimizer, required

from distbelief.utils.messaging import send_message, GSMessageCode, GradientMessageListener
from distbelief.utils.serialization import ravel_model_params, update_model_params

_LOGGER = logging.getLogger(__name__)


class GradientListener(GradientMessageListener):
    """DownpourListener"""

    def __init__(self, model, queue):
        super().__init__(model)
        self.lr = 0.05
        self.queue = queue

    def receive(self, sender, message_code, gradient_version, parameter, ):
        """receive parameter updates from the server and reflect them into the client's model."""
        _LOGGER.info("Processing message: {}, version: {}, lr: {}".format(message_code.name, gradient_version, self.lr))
        if message_code == GSMessageCode.GradientUpdate:
            update_model_params(self.model, parameter, self.lr)
            self.queue.put(gradient_version)


class GradientSGD(Optimizer):
    """GradientSGD"""

    def __init__(self, params, lr=required, n_push=0, n_pull=0, model=required):
        """__init__

        :param params:
        :param lr:
        :param freq:
        :param model:
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        print('I am node rank:%d' % dist.get_rank())
        defaults = dict(lr=lr, )
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.model = model
        # this sets the initial model parameters
        # send_message(MessageCode.ParameterUpdate, ravel_model_params(self.model))
        self.idx = 0
        self.version = 0
        self.queue = Queue(maxsize=1)
        self.listener = GradientListener(self.model, self.queue)
        self.listener.start()

        super(GradientSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # increase version No.
        self.version += 1
        if dist.get_rank() == 1:
            time.sleep(0.1)

        # get the lr
        lr = self.param_groups[0]['lr']
        self.listener.lr = lr

        # keep track of accumulated gradients so that we can send 
        gradients = ravel_model_params(self.model, grads=True)
        send_message(GSMessageCode.GradientUpdate, gradients, dst=0, gradient_version=self.version)

        # reset gradient version
        self.version = self.queue.get()

        self.idx += 1
        return loss
