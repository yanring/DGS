import logging
import os
import queue
import socket
import time
from enum import Enum
from multiprocessing.managers import BaseManager
from threading import Thread

import torch
import torch.distributed as dist

from distbelief.utils.serialization import ravel_model_params

_LOGGER = logging.getLogger(__name__)

isCUDA = 0
manager = None


def tail(filename):
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.01)
                continue
            if dist.get_rank() == 0:
                print('delay:', time.time() - os.stat(filename).st_mtime)
            yield int(line)


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
        self.running = True
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
        while self.running:
            _LOGGER.info("Polling for message...")
            dist.recv(tensor=self.m_parameter)
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])


a1 = queue.Queue()
b1 = queue.Queue()
a2 = queue.Queue()
b2 = queue.Queue()
a3 = queue.Queue()
b3 = queue.Queue()
a4 = queue.Queue()
b4 = queue.Queue()
a5 = queue.Queue()
b5 = queue.Queue()
a6 = queue.Queue()
b6 = queue.Queue()
a7 = queue.Queue()
b7 = queue.Queue()
a8 = queue.Queue()
b8 = queue.Queue()


def rta1():
    return a1


def rtb1():
    return b1


def rta2():
    return a2


def rtb2():
    return b2


def rta3():
    return a3


def rtb3():
    return b3


def rta4():
    return a4


def rtb4():
    return b4


def rta5():
    return a5


def rtb5():
    return b5


def rta6():
    return a6


def rtb6():
    return b6


def rta7():
    return a7


def rtb7():
    return b7


def rta8():
    return a8


def rtb8():
    return b8


class GradientMessageListener(Thread):
    """MessageListener

    base class for message listeners, extends pythons threading Thread
    """

    def __init__(self, model_size, source=0):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        # self.model = model
        self.source = source
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(model_size + 4).double()
        self.cached_stamp = 0
        self.size_filename = None
        self.manager = None

        if dist.get_rank() == 0 and self.source == 1:
            self.init_server_queue_manager()
        elif dist.get_rank() > 0:
            self.recv_queue, self.send_queue = self.init_worker_queue_manager()
        super(GradientMessageListener, self).__init__()

    def receive(self, sender, message_code, gradient_version, lr, parameter):
        """receive

        :param lr:
        :param gradient_version:
        :param sender: rank id of the sender
        :param message_code: Enum code
        :param parameter: the data payload
        """
        raise NotImplementedError()

    def run(self):
        # for sparse gradient transmission
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for sparse message...")
            # for size in tail(self.size_filename):
            while True:
                size = QueueManager.get_size(self.source)
                # if dist.get_rank() == 0:
                # print('RECEIVING MESSAGE %dto%d.size:%d,' % (
                #     self.source, dist.get_rank(), size))
                self.m_parameter = torch.zeros(size + 4).double()
                try:
                    sender = dist.recv(tensor=self.m_parameter, src=self.source)
                except Exception as e:
                    # print('Exception :', e)
                    raise e
                    time.sleep(0.5)
                    continue
                self.m_parameter = self.m_parameter
                # if dist.get_rank() == 0:
                #     print('run',self.m_parameter[int(len(self.m_parameter) / 2)-3:int(len(self.m_parameter) / 2)+2],self.m_parameter[int(len(self.m_parameter) / 2)-3:int(len(self.m_parameter) / 2)+2].long())
                self.receive(int(self.m_parameter[0].item()),
                             GSMessageCode(self.m_parameter[1].item()),
                             int(self.m_parameter[2].item()),
                             float(self.m_parameter[3].item()),
                             self.m_parameter[4:])

    def init_server_queue_manager(self):

        QueueManager.register('from0to%d' % 1, callable=rta1)
        QueueManager.register('from%dto0' % 1, callable=rtb1)
        QueueManager.register('from0to%d' % 2, callable=rta2)
        QueueManager.register('from%dto0' % 2, callable=rtb2)
        QueueManager.register('from0to%d' % 3, callable=rta3)
        QueueManager.register('from%dto0' % 3, callable=rtb3)
        QueueManager.register('from0to%d' % 4, callable=rta4)
        QueueManager.register('from%dto0' % 4, callable=rtb4)
        QueueManager.register('from0to%d' % 5, callable=rta5)
        QueueManager.register('from%dto0' % 5, callable=rtb5)
        QueueManager.register('from0to%d' % 6, callable=rta6)
        QueueManager.register('from%dto0' % 6, callable=rtb6)
        QueueManager.register('from0to%d' % 7, callable=rta7)
        QueueManager.register('from%dto0' % 7, callable=rtb7)
        QueueManager.register('from0to%d' % 8, callable=rta8)
        QueueManager.register('from%dto0' % 8, callable=rtb8)
        self.manager = QueueManager(address=('', 5000), authkey=b'abc')
        QueueManager.send_queue_list.append(0)
        QueueManager.recv_queue_list.append(0)
        QueueManager.manager = self.manager
        self.manager.start()
        for i in range(1, dist.get_world_size()):
            send_queue = eval('self.manager.from0to%d' % i)()
            QueueManager.send_queue_list.append(send_queue)
            recv_queue = eval('self.manager.from%dto0' % i)()
            QueueManager.recv_queue_list.append(recv_queue)
        pass

    def init_worker_queue_manager(self):
        # time.sleep(1)
        QueueManager.register('from0to%d' % dist.get_rank())
        QueueManager.register('from%dto0' % dist.get_rank())
        time.sleep(10)
        # self.manager = QueueManager(address=(socket.gethostbyname('localhost'), 5000), authkey=b'abc')
        if socket.gethostname() == 'yan-pc' or socket.gethostname() == 'yrx-MS-7A93' or 'ubuntu' in socket.gethostname():
            print('queue init in 522')
            # self.manager = QueueManager(address=('172.18.166.108', 5000), authkey=b'abc')
            self.manager = QueueManager(address=('192.168.3.100', 5000), authkey=b'abc')
        else:
            time.sleep(10)
            print('queue init in th')
            self.manager = QueueManager(address=('89.72.3.16', 5000), authkey=b'abc')
        try:
            self.manager.connect()
        except Exception as e:
            print(e)
            time.sleep(10)
            self.manager.connect()
        send_queue = eval('self.manager.from%dto0' % dist.get_rank())()
        QueueManager.send_queue_list.append(send_queue)
        recv_queue = eval('self.manager.from0to%d' % dist.get_rank())()
        QueueManager.recv_queue_list.append(recv_queue)
        QueueManager.manager = self.manager
        return recv_queue, send_queue


# class GradientServer(GradientMessageListener):
#     def __init__(self, share_tensor, shared_queue_recv, shared_queue_send, model_size, source=0):
#         self.running = True
#         print('Creating GradientMessageListener %d' % source)
#         self.shared_queue_recv = shared_queue_recv
#         self.shared_queue_send = shared_queue_send
#         self.shared_gradient = share_tensor
#         # self.shared_gradient.share_memory_()
#         self._start = None
#         self._end = None
#         super().__init__(model_size, source)
#
#     def receive(self, sender, message_code, gradient_version, lr, parameter):
#         # put gradient update to queue
#         if message_code == GSMessageCode.SparseGradientUpdate:
#             self.shared_gradient.copy_(unravel_sparse_gradient(parameter))
#         else:
#             self.shared_gradient.copy_(parameter)
#         self.shared_queue_recv.put([sender, message_code, gradient_version, lr])
#         # self.shared_queue_recv.put([sender, message_code, gradient_version, parameter])
#
#     def send_message(self, payload, message_code, gradient_version=None, lr=None):
#         """Sends a message to a destination
#         Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
#         """
#         _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
#         m_parameter = torch.Tensor([dist.get_rank(), message_code.value, gradient_version, lr])
#         # payload = self.shared_gradient
#         size = str(payload.numel())
#         payload = torch.cat((m_parameter, payload))
#         QueueManager.put_size(self.source, size)
#         if dist.get_rank() == 0:
#             print('%s SENDING MESSAGE %s gradient_version %d, %dto%d.size:%d' % (
#                 str(time.time()), message_code, gradient_version, dist.get_rank(), self.source, payload.numel()))
#         dist.isend(tensor=payload, dst=self.source)
#
#     def run(self):
#         # for sparse gradient transmission
#         _LOGGER.info("Started Running!")
#         # print('get message from %d' % self.source)
#         send_info = self.shared_queue_send.get()
#         self.send_message(send_info[0], send_info[1], gradient_version=send_info[2], lr=0.1)
#         while self.running:
#             _LOGGER.info("Polling for sparse message...")
#             # for size in tail(self.size_filename):
#             while True:
#                 size = QueueManager.get_size(self.source)
#                 self._start = time.time()
#                 if dist.get_rank() == 0:
#                     print('RECEIVING MESSAGE %dto%d.size:%d,' % (
#                         self.source, dist.get_rank(), size))
#                 self.m_parameter = torch.zeros(size + 4)
#                 try:
#                     sender = dist.recv(tensor=self.m_parameter, src=self.source)
#                 except Exception as e:
#                     print('Exception :', e)
#                     time.sleep(0.5)
#                     continue
#                 # self.m_parameter = self.m_parameter.cuda()
#                 # print('start',time.time())s
#
#                 self.receive(int(self.m_parameter[0].item()),
#                              GSMessageCode(self.m_parameter[1].item()),
#                              int(self.m_parameter[2].item()),
#                              float(self.m_parameter[3].item()),
#                              self.m_parameter[4:])
#
#                 send_info = self.shared_queue_send.get()
#                 self.send_message(send_info[0], send_info[1], gradient_version=send_info[2], lr=send_info[3])
#                 self._end = time.time()
#                 print('server cal+messaging cost time : %f' % (self._end - self._start))


class QueueManager(BaseManager):
    manager = None
    send_queue_list = []
    recv_queue_list = []

    @classmethod
    def get_manager(cls):
        return cls.manager

    @classmethod
    def get_size(cls, opposite):
        recv_queue = cls.recv_queue_list[opposite]
        # exec('recv_queue = cls.manager.from%dto%d()' % (source, target))
        res = None
        try:
            res = recv_queue.get(timeout=400)
        except queue.Empty:
            print('task queue is empty')
        # print('RECV ', res, type(recv_queue), recv_queue)
        return int(res)

    @classmethod
    def put_size(cls, opposite, size):
        # send_queue = None
        # exec('send_queue = cls.manager.from%dto%d()' % (source, target))
        send_queue = cls.send_queue_list[opposite]
        # print('SEND ', type(send_queue), size, send_queue)
        send_queue.put(size)


def send_message(message_code, payload, dst=0, gradient_version=None, lr=0.1):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    # _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    m_parameter = torch.Tensor([dist.get_rank(), message_code.value, gradient_version, lr])
    # print(m_parameter.size(), payload.size())
    if payload.is_cuda:
        payload = payload.cpu()
    size = str(payload.numel())
    payload = torch.cat((m_parameter.double(), payload.double()))
    if dist.get_rank() == 0 and dist.get_world_size() > 5:
        print('%s SENDING MESSAGE %s gradient_version %d, %dto%d.size:%d' % (
            str(time.time()), message_code, gradient_version, dist.get_rank(), dst, payload.numel()))
    # with open('%dto%d.size' % (dist.get_rank(), dst), 'a') as f:
    #     f.write(size)
    QueueManager.put_size(dst, size)
    dist.isend(tensor=payload, dst=dst)
