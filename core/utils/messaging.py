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

from core.utils.serialization import ravel_model_params

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
a9 = queue.Queue()
b9 = queue.Queue()
a10 = queue.Queue()
b10 = queue.Queue()
a11 = queue.Queue()
b11 = queue.Queue()
a12 = queue.Queue()
b12 = queue.Queue()
a13 = queue.Queue()
b13 = queue.Queue()
a14 = queue.Queue()
b14 = queue.Queue()
a15 = queue.Queue()
b15 = queue.Queue()
a16 = queue.Queue()
b16 = queue.Queue()
a17 = queue.Queue()
b17 = queue.Queue()
a18 = queue.Queue()
b18 = queue.Queue()
a19 = queue.Queue()
b19 = queue.Queue()
a20 = queue.Queue()
b20 = queue.Queue()
a21 = queue.Queue()
b21 = queue.Queue()
a22 = queue.Queue()
b22 = queue.Queue()
a23 = queue.Queue()
b23 = queue.Queue()
a24 = queue.Queue()
b24 = queue.Queue()
a25 = queue.Queue()
b25 = queue.Queue()
a26 = queue.Queue()
b26 = queue.Queue()
a27 = queue.Queue()
b27 = queue.Queue()
a28 = queue.Queue()
b28 = queue.Queue()
a29 = queue.Queue()
b29 = queue.Queue()
a30 = queue.Queue()
b30 = queue.Queue()
a31 = queue.Queue()
b31 = queue.Queue()
a32 = queue.Queue()
b32 = queue.Queue()
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
def rta9():
    return a9
def rtb9():
    return b9
def rta10():
    return a10
def rtb10():
    return b10
def rta11():
    return a11
def rtb11():
    return b11
def rta12():
    return a12
def rtb12():
    return b12
def rta13():
    return a13
def rtb13():
    return b13
def rta14():
    return a14
def rtb14():
    return b14
def rta15():
    return a15
def rtb15():
    return b15
def rta16():
    return a16
def rtb16():
    return b16


def rta17():
    return a17


def rtb17():
    return b17


def rta18():
    return a18


def rtb18():
    return b18


def rta19():
    return a19


def rtb19():
    return b19


def rta20():
    return a20


def rtb20():
    return b20


def rta21():
    return a21


def rtb21():
    return b21


def rta22():
    return a22


def rtb22():
    return b22


def rta23():
    return a23


def rtb23():
    return b23


def rta24():
    return a24


def rtb24():
    return b24


def rta25():
    return a25


def rtb25():
    return b25


def rta26():
    return a26


def rtb26():
    return b26


def rta27():
    return a27


def rtb27():
    return b27


def rta28():
    return a28


def rtb28():
    return b28


def rta29():
    return a29


def rtb29():
    return b29


def rta30():
    return a30


def rtb30():
    return b30


def rta31():
    return a31


def rtb31():
    return b31


def rta32():
    return a32


def rtb32():
    return b32


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

        QueueManager.register('from0to1', callable=rta1)
        QueueManager.register('from1to0', callable=rtb1)
        QueueManager.register('from0to2', callable=rta2)
        QueueManager.register('from2to0', callable=rtb2)
        QueueManager.register('from0to3', callable=rta3)
        QueueManager.register('from3to0', callable=rtb3)
        QueueManager.register('from0to4', callable=rta4)
        QueueManager.register('from4to0', callable=rtb4)
        QueueManager.register('from0to5', callable=rta5)
        QueueManager.register('from5to0', callable=rtb5)
        QueueManager.register('from0to6', callable=rta6)
        QueueManager.register('from6to0', callable=rtb6)
        QueueManager.register('from0to7', callable=rta7)
        QueueManager.register('from7to0', callable=rtb7)
        QueueManager.register('from0to8', callable=rta8)
        QueueManager.register('from8to0', callable=rtb8)
        QueueManager.register('from0to9', callable=rta9)
        QueueManager.register('from9to0', callable=rtb9)
        QueueManager.register('from0to10', callable=rta10)
        QueueManager.register('from10to0', callable=rtb10)
        QueueManager.register('from0to11', callable=rta11)
        QueueManager.register('from11to0', callable=rtb11)
        QueueManager.register('from0to12', callable=rta12)
        QueueManager.register('from12to0', callable=rtb12)
        QueueManager.register('from0to13', callable=rta13)
        QueueManager.register('from13to0', callable=rtb13)
        QueueManager.register('from0to14', callable=rta14)
        QueueManager.register('from14to0', callable=rtb14)
        QueueManager.register('from0to15', callable=rta15)
        QueueManager.register('from15to0', callable=rtb15)
        QueueManager.register('from0to16', callable=rta16)
        QueueManager.register('from16to0', callable=rtb16)
        QueueManager.register('from0to17', callable=rta17)
        QueueManager.register('from17to0', callable=rtb17)
        QueueManager.register('from0to18', callable=rta18)
        QueueManager.register('from18to0', callable=rtb18)
        QueueManager.register('from0to19', callable=rta19)
        QueueManager.register('from19to0', callable=rtb19)
        QueueManager.register('from0to20', callable=rta20)
        QueueManager.register('from20to0', callable=rtb20)
        QueueManager.register('from0to21', callable=rta21)
        QueueManager.register('from21to0', callable=rtb21)
        QueueManager.register('from0to22', callable=rta22)
        QueueManager.register('from22to0', callable=rtb22)
        QueueManager.register('from0to23', callable=rta23)
        QueueManager.register('from23to0', callable=rtb23)
        QueueManager.register('from0to24', callable=rta24)
        QueueManager.register('from24to0', callable=rtb24)
        QueueManager.register('from0to25', callable=rta25)
        QueueManager.register('from25to0', callable=rtb25)
        QueueManager.register('from0to26', callable=rta26)
        QueueManager.register('from26to0', callable=rtb26)
        QueueManager.register('from0to27', callable=rta27)
        QueueManager.register('from27to0', callable=rtb27)
        QueueManager.register('from0to28', callable=rta28)
        QueueManager.register('from28to0', callable=rtb28)
        QueueManager.register('from0to29', callable=rta29)
        QueueManager.register('from29to0', callable=rtb29)
        QueueManager.register('from0to30', callable=rta30)
        QueueManager.register('from30to0', callable=rtb30)
        QueueManager.register('from0to31', callable=rta31)
        QueueManager.register('from31to0', callable=rtb31)
        QueueManager.register('from0to32', callable=rta32)
        QueueManager.register('from32to0', callable=rtb32)

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
            self.manager = QueueManager(address=('10.20.9.2', 5000), authkey=b'abc')
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
            res = recv_queue.get(timeout=4000)
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
    if dist.get_rank() == 0 and gradient_version % 100 == 0:
        print('%s SENDING MESSAGE %s gradient_version %d, %dto%d.size:%d' % (
            str(time.time()), message_code, gradient_version, dist.get_rank(), dst, payload.numel()))
    # with open('%dto%d.size' % (dist.get_rank(), dst), 'a') as f:
    #     f.write(size)
    QueueManager.put_size(dst, size)
    dist.send(tensor=payload, dst=dst)
