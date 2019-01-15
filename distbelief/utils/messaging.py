import logging
import torch
import torch.distributed as dist
from enum import Enum
from threading import Thread

from distbelief.utils.serialization import ravel_model_params

_LOGGER = logging.getLogger(__name__)


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

    def __init__(self, model):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        _LOGGER.info("Setting m_parameter")
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 5)
        print('Tensor size : %d' % self.m_parameter.numel())
        super(GradientMessageListener, self).__init__()

    def receive(self, sender, message_code, gradient_version, trigger, fast_flag, parameter):
        """receive

        :param fast_flag:
        :param trigger:
        :param gradient_version:
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
                         GSMessageCode(self.m_parameter[1].item()),
                         int(self.m_parameter[2].item()),
                         int(self.m_parameter[3].item()),
                         int(self.m_parameter[4].item()),
                         self.m_parameter[5:])


class GradientMessageSender(Thread):
    """MessageListener

    base class for message listeners, extends pythons threading Thread
    """

    def __init__(self, queue):
        """__init__

        :param queue: queue for gradient
        """
        self.queue = queue
        super(GradientMessageSender, self).__init__()

    def send_message(self):
        """Sends a message to a destination
        Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
        """
        message_code, payload, dst, gradient_version, trigger, fast_flag = self.queue.get()
        _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
        if 1:
            if message_code.value == GSMessageCode.GradientUpdate:
                print("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
            m_parameter = torch.Tensor([dist.get_rank(), message_code.value, gradient_version, trigger, fast_flag])

        m_parameter = torch.cat((m_parameter, payload))
        dist.isend(tensor=m_parameter, dst=dst)
        # print('send version %d'%gradient_version)

    def run(self):
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            self.send_message()


def send_message(message_code, payload, dst=0, gradient_version=None, trigger=0, fast_flag=0):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    # print("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))

    m_parameter = torch.Tensor([dist.get_rank(), message_code.value, gradient_version, trigger, fast_flag])

    m_parameter = torch.cat((m_parameter, payload))
    dist.isend(tensor=m_parameter, dst=dst)
