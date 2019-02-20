import time

import logging
import os
import torch
import torch.distributed as dist
from enum import Enum
from threading import Thread

from distbelief.utils.serialization import ravel_model_params

_LOGGER = logging.getLogger(__name__)

isCUDA = 0


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3


class GSMessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    GradientRequest = 0
    GradientUpdate = 1
    # ParameterUpdate = 2
    EvaluateParams = 3
    ModelRequest = 4
    ModelUpdate = 5
    SparseGradientUpdate = 6


class ModelSize(Enum):
    """Different model size"""
    AlexNet = 2472266
    ResNet18 = 11173962


class MessageListener(Thread):
    """MessageListener
   
    base class for message listeners, extends pythons threading Thread
    """

    def __init__(self, model):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        self.model = model
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)
        super(MessageListener, self).__init__()

    def receive(self, sender, message_code, parameter):
        """receive

        :param sender: rank id of the sender
        :param message_code: Enum code 
        :param parameter: the data payload
        """
        raise NotImplementedError()

    def run(self):
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for message...")
            dist.recv(tensor=self.m_parameter)
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])


class GradientMessageListener(Thread):
    """MessageListener

    base class for message listeners, extends pythons threading Thread
    """

    def __init__(self, model, source=0):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        self.model = model
        self.source = source
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model, cuda=True).numel() + 3)
        self.cached_stamp = 0
        self.size_filename = None
        super(GradientMessageListener, self).__init__()

    def receive(self, sender, message_code, gradient_version, parameter):
        """receive

        :param gradient_version:
        :param sender: rank id of the sender
        :param message_code: Enum code
        :param parameter: the data payload
        """
        raise NotImplementedError()

    # def run(self):
    #     # for dense gradient transmission
    #     _LOGGER.info("Started Running!")
    #     self.running = True
    #     while self.running:
    #         _LOGGER.info("Polling for dense message...")
    #         # dist.recv(tensor=self.m_parameter)
    #         # i = torch.LongTensor([[0, 1, 1],
    #         #                       [2, 0, 2]])
    #         # v = torch.FloatTensor([3, 4, 5])
    #         # m_parameter = torch.sparse.FloatTensor(i, v, torch.Size([10, 3]))
    #         sender = dist.recv(tensor=self.m_parameter)
    #         # print(m_parameter)
    #         self.receive(int(self.m_parameter[0].item()),
    #                      GSMessageCode(self.m_parameter[1].item()),
    #                      int(self.m_parameter[2].item()),
    #                      self.m_parameter[3:])

    def run(self):
        # for sparse gradient transmission
        _LOGGER.info("Started Running!")
        self.running = True
        self.size_filename = '%dto%d.size' % (self.source, dist.get_rank())
        while not os.path.exists(self.size_filename):
            time.sleep(0.5)
        while self.running:
            _LOGGER.info("Polling for sparse message...")
            while os.stat(self.size_filename).st_mtime == self.cached_stamp or os.stat(
                    self.size_filename).st_mtime - self.cached_stamp < 0.01:
                time.sleep(0.005)
            time.sleep(0.01)
            with open(self.size_filename, 'r') as f:
                try:
                    size = int(float(f.read().strip()))
                    if dist.get_rank() == 0:
                        print('RECEIVING MESSAGE %dto%d.size:%d, changed time : %s' % (
                            self.source, dist.get_rank(), size, str(self.cached_stamp)))
                except Exception as _:
                    time.sleep(0.05)
                    size = int(float(f.read().strip()))
                    if dist.get_rank() == 0:
                        print('RECEIVING MESSAGE %dto%d.size:%d, changed time : %s' % (
                            self.source, dist.get_rank(), size, str(self.cached_stamp)))
                    self.m_parameter = torch.zeros(size + 3)
            try:
                sender = dist.recv(tensor=self.m_parameter, src=self.source)
            except Exception as e:
                print('Exception :', e)
                continue
            self.cached_stamp = os.stat(self.size_filename).st_mtime
            if dist.get_rank() >= 0:
                self.m_parameter = self.m_parameter.cuda()
            self.receive(int(self.m_parameter[0].item()),
                         GSMessageCode(self.m_parameter[1].item()),
                         int(self.m_parameter[2].item()),
                         self.m_parameter[3:])


def send_message(message_code, payload, dst=0, gradient_version=None):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value, gradient_version])
    # print(m_parameter.size(), payload.size())
    if payload.is_cuda:
        payload = payload.cpu()
    m_parameter = torch.cat((m_parameter, payload))
    if dist.get_rank() == 0:
        print('%s SENDING MESSAGE %s gradient_version %d, %dto%d.size:%d' % (
            str(time.time()), message_code, gradient_version, dist.get_rank(), dst, payload.numel()))
    size = str(payload.numel())
    with open('%dto%d.size' % (dist.get_rank(), dst), 'w') as f:
        f.write(size)
    dist.isend(tensor=m_parameter, dst=dst)

#
# def send_sparse_gradient(net, dst=0, gradient_version=None):
#     _LOGGER.info("SENDING SPARSE MESSAGE: {} RANK: {}".format('send_sparse_message', dist.get_rank()))
#     # size, sparse_gradient = ravel_sparse_gradient(net)
#     # send_message(GSMessageCode.SparseSize, size, dst, gradient_version)
#     # time.sleep(0.1)
#     raise Exception('')
#     # send_message(GSMessageCode.SparseIndex, index, dst, gradient_version)
#     # send_message(GSMessageCode.SparseGradientUpdate, sparse_gradient, dst, gradient_version)
#     # dist.isend(tensor=m_parameter, dst=dst)
