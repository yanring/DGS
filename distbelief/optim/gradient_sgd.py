import logging
import os
import sys
import threading
import time
from datetime import datetime
from queue import Queue

import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required

from distbelief.utils.messaging import send_message, GSMessageCode, GradientMessageListener
from distbelief.utils.serialization import ravel_model_params, update_model_params, unravel_model_params, \
    ravel_sparse_gradient, unravel_sparse_gradient, worker_gradient_executor, DGC, Aji

WORKPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(WORKPATH)
print(WORKPATH)
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
            # print('4',parameter.sum())

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

    def __init__(self, params, lr=required, model=required, momentum=None, weight_decay=0, args=None):
        """
        :param params:
        :param lr:
        :param n_push:
        :param n_pull:
        :param model:
        """
        print('in my optimizer ')
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, )
        self.model = model
        self.filter_gradient = ravel_model_params(model)
        self.momentum = momentum
        self.v_kt = self.filter_gradient.clone().zero_()
        self.u_kt = self.filter_gradient.clone().zero_()
        self.idx = 0
        self.version = 0
        self.queue = Queue(maxsize=1)
        if not args.no_distributed and args.rank > 0:
            dist.init_process_group('tcp', init_method='file://%s/sharedfile' % WORKPATH, group_name='mygroup',
                                    world_size=args.world_size, rank=args.rank)
            print('I am node rank:%d' % dist.get_rank())
            self.listener = GradientListener(model, self.queue)
            self.listener.start()
        self.tmp = 0
        self.compress_ratio = None
        self.weight_decay = weight_decay
        print('weight_decay', self.weight_decay, 'lr', lr, 'momentum', self.momentum)
        self.args = args
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
        if not self.args.no_distributed and not self.listener.flag:
            while not self.args.no_distributed and not self.listener.flag:
                print('wait for server')
                time.sleep(1)
            return loss

        # get the lr
        if self.args.rank == 1:
            lr = self.param_groups[0]['lr']
            # lr = 0.2
        else:
            if self.tmp != self.listener.lr:
                print('lr from %f to %f' % (self.tmp, self.listener.lr))
                self.tmp = self.listener.lr
                self.param_groups[0]['lr'] = self.tmp
            lr = self.param_groups[0]['lr']
        # print(lr)
        # lr = self.param_groups[0]['lr']
        # keep track of accumulated gradients so that we can send
        # ASYNC
        if self.args.mode == 'asgd':
            print('Running asgd')

            self.filter_gradient = ravel_model_params(self.model, grads=True, cuda=True).mul_(lr)
            send_message(GSMessageCode.GradientUpdate, self.filter_gradient, dst=0,
                         gradient_version=self.listener.version + 1)
            self.version = self.queue.get()
            self.idx += 1
            return loss
        elif self.args.mode == 'gradient_sgd':
            if self.version < 5:
                print('Running gradient_sgd')
            if self.version < 900:
                weight_decay = 0
            else:
                weight_decay = self.weight_decay
            raveled_gradients = worker_gradient_executor(self.model, self.filter_gradient, self.u_kt, self.v_kt,
                                                         # rate=0.04 * (lr / self.args.lr) / (self.args.world_size - 1),
                                                         rate=0.01,
                                                         lr=lr, momentum=self.momentum, weight_decay=weight_decay)
            # print(1,raveled_gradients.sum())
            sparse_gradient = ravel_sparse_gradient(raveled_gradients)

        elif self.args.mode == 'dgc':
            if self.version < 5:
                print('Running dgc')
            raveled_gradients = DGC(self.model, self.filter_gradient, self.u_kt, self.v_kt,
                                    rate=0.01,
                                    # rate=self.compress_ratio,
                                    lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
            sparse_gradient = ravel_sparse_gradient(raveled_gradients)
        elif self.args.mode == 'aji':
            if self.version < 5:
                print('Running aji ', self.version)
            raveled_gradients = Aji(self.model, self.filter_gradient, self.u_kt, self.v_kt,
                                    rate=0.01,
                                    lr=lr, momentum=self.momentum)
            sparse_gradient = ravel_sparse_gradient(raveled_gradients)
        elif self.args.mode == 'sgd':
            if self.version < 5:
                print('Running sgd')
            weight_decay = self.weight_decay
            momentum = self.momentum
            dampening = 0
            nesterov = 0
            g = ravel_model_params(self.model, grads=True)
            p = ravel_model_params(self.model, grads=False)
            g.add_(weight_decay, p)
            self.u_kt.mul_(momentum).add_(g)
            raveled_gradients = self.u_kt.mul(lr)
        else:
            raise Exception('no optimizer')
        # reset gradient version
        if self.args.no_distributed:
            # parameter = unravel_sparse_gradient(sparse_gradient).cuda()
            update_model_params(self.model, raveled_gradients, 1)
            # self.version = gradient_version
            self.queue.put(self.idx)
        else:
            send_message(GSMessageCode.SparseGradientUpdate, sparse_gradient, dst=0,
                         gradient_version=self.listener.version + 1, lr=lr)
        self.version = self.queue.get()
        self.idx += 1
        return loss