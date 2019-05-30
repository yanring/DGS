import logging
import os
import sys
import threading
import time
import torch.distributed as dist
from datetime import datetime
from queue import Queue
from torch.optim.optimizer import Optimizer, required

from distbelief.utils.messaging import send_message, GSMessageCode, GradientMessageListener
from distbelief.utils.serialization import ravel_model_params, update_model_params, unravel_model_params, \
    ravel_sparse_gradient, unravel_sparse_gradient, worker_gradient_executor

WORKPATH = os.path.abspath(os.path.dirname(os.path.dirname('main.py')))
sys.path.append(WORKPATH)

_LOGGER = logging.getLogger(__name__)
lock = threading.Lock()


class GradientListener(GradientMessageListener):
    """DownpourListener"""

    def __init__(self, model, queue):
        super(GradientListener, self).__init__(ravel_model_params(model).numel())
        self.lr = 0.05
        self.queue = queue
        self.version = 0
        self.model = model
        self.flag = False

    def receive(self, sender, message_code, gradient_version, lr, parameter):
        """receive parameter updates from the server and reflect them into the client's model."""
        _LOGGER.info("Processing message: {}, version: {}, lr: {}".format(message_code.name, gradient_version, self.lr))
        # print("Processing message: {}, version: {}, lr: {}".format(message_code.name, gradient_version, lr))
        self.lr = lr
        if message_code == GSMessageCode.GradientUpdate:
            update_model_params(self.model, parameter, -1)
            self.version = gradient_version
            self.queue.put(gradient_version)
        elif message_code == GSMessageCode.SparseGradientUpdate:
            parameter = unravel_sparse_gradient(parameter).cuda()
            update_model_params(self.model, parameter, -1)
            self.version = gradient_version
            self.queue.put(gradient_version)
        elif message_code == GSMessageCode.ModelRequest:
            model = ravel_model_params(self.model, grads=False)
            send_message(GSMessageCode.ModelUpdate, model, dst=0, gradient_version=0)
            print('send model to server')
        elif message_code == GSMessageCode.ModelUpdate:
            print('sync model!', gradient_version, ' ', datetime.now(), ' synced model :', parameter.sum())
            unravel_model_params(self.model, parameter)
            self.version = gradient_version
            self.flag = True
            # TODO change back
            if self.version > 1:
                self.queue.put(gradient_version)
            # lock.release()


class GradientSGD(Optimizer):
    """GradientSGD"""

    def __init__(self, params, lr=required, model=required, momentum=None):
        """
        :param params:
        :param lr:
        :param n_push:
        :param n_pull:
        :param model:
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        print('I am node rank:%d' % dist.get_rank())
        defaults = dict(lr=lr, )
        self.model = model
        self.filter_gradient = ravel_model_params(model)
        self.momentum = momentum
        self.v_kt = self.filter_gradient.clone().zero_()
        self.u_kt = self.filter_gradient.clone().zero_()
        self.idx = 0
        self.version = 0
        self.queue = Queue(maxsize=1)
        self.listener = GradientListener(model, self.queue)
        self.listener.start()
        self.tmp = 0
        self.compress_ratio = None
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

        if not self.listener.flag:
            time.sleep(1)
            return loss

        # get the lr
        if dist.get_rank() == 1:
            lr = self.param_groups[0]['lr']
            # lr = 0.2
        else:
            if self.tmp != self.listener.lr:
                print('lr from %f to %f' % (self.tmp, self.listener.lr))
                self.tmp = self.listener.lr
            lr = self.listener.lr
            # print(lr)

        # keep track of accumulated gradients so that we can send
        # ASYNC
        # self.filter_gradient = ravel_model_params(self.model, grads=True, cuda=True).mul_(lr)
        # send_message(GSMessageCode.GradientUpdate, self.filter_gradient, dst=0,
        #              gradient_version=self.listener.version + 1)

        # COMPRESSION
        # if self.version < 781 / 4 * 3:
        #     rate = 0.01
        # elif self.version < 7810 / 4 * 2:
        #     rate = 0.01
        # else:
        #     rate = 0.001
        raveled_gradients = worker_gradient_executor(self.model, self.filter_gradient, self.u_kt, self.v_kt,
                                                     rate=0.1 * lr,
                                                     lr=lr, momentum=self.momentum)
        sparse_gradient = ravel_sparse_gradient(raveled_gradients)
        send_message(GSMessageCode.SparseGradientUpdate, sparse_gradient, dst=0,
                     gradient_version=self.listener.version + 1, lr=lr)

        # reset gradient version
        self.version = self.queue.get()
        self.idx += 1
        return loss
