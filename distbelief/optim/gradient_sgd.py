import time

import logging
import threading
import torch
from torch.optim.optimizer import Optimizer, required
from distbelief.utils.serialization import ravel_model_params, update_model_params
from distbelief.utils.messaging import MessageListener, send_message, GSMessageCode, GradientMessageListener

_LOGGER = logging.getLogger(__name__)

lock = threading.Lock()


class GradientListener(GradientMessageListener):
    """DownpourListener"""

    def __init__(self, model):
        super().__init__(model)
        self.lr = 0.1

    def receive(self, sender, message_code, gradient_version, parameter, ):
        """receive parameter updates from the server and reflect them into the client's model."""
        _LOGGER.info("Processing message: {}, version: {}".format(message_code.name, gradient_version))
        if message_code == GSMessageCode.GradientUpdate:
            update_model_params(self.model, parameter, self.lr)
            while True:
                if lock.locked():
                    try:
                        lock.release()
                        break
                    except Exception as e:
                        print(e)


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

        defaults = dict(lr=lr, )
        self.accumulated_gradients = torch.zeros(ravel_model_params(model).size())
        self.model = model
        # this sets the initial model parameters
        # send_message(MessageCode.ParameterUpdate, ravel_model_params(self.model))
        self.idx = 0
        self.version = 0

        self.listener = GradientListener(self.model)
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

        # send parameter request every N iterations
        # if self.idx % self.n_pull == 0:
        #     send_message(MessageCode.ParameterRequest, self.accumulated_gradients)  # dummy val

        # get the lr
        lr = self.param_groups[0]['lr']
        # self.listener.set_lr(lr)
        # keep track of accumulated gradients so that we can send 
        gradients = ravel_model_params(self.model, grads=True)
        send_message(GSMessageCode.GradientUpdate, gradients, dst=0, gradient_version=self.version)
        # print("worker send gradient to server")
        lock.acquire()
        lock.acquire()
        lock.release()
        # send gradient update every N iterations
        # if self.idx % self.n_push == 0:
        #     send_message(MessageCode.GradientUpdate, self.accumulated_gradients)  # send gradients to the server
        #     self.accumulated_gradients.zero_()

        # internal sgd update

        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data
        #         p.data.add_(-group['lr'], d_p)

        self.idx += 1
        return loss
