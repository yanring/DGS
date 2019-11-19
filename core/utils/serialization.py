import time

import torch

from core.utils import constant

current_model_size = None


def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if next(model.parameters()).is_cuda:
        m_parameter = torch.Tensor([0]).cuda()
    else:
        m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]


def unravel_model_params(model, parameter_update):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from parameter_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        parameter.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel


def update_model_params(model, parameter_update, lr):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() and adding the gradient in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from parameter_update
    for parameter in model.parameters():
        numel = parameter.data.numel()
        size = parameter.data.size()
        # print(parameter.data.device,parameter_update.device)
        parameter.data.add_(-lr, parameter_update[current_index:current_index + numel].view(size))
        current_index += numel


def worker_gradient_executor(net, payload, u_kt, v_kt, rate=0.01, lr=0.1, momentum=None, weight_decay=0):
    """
    :param momentum:
    :param lr:
    :param v_kt:
    :param payload:
    :param u_kt:
    :param net: model
    :param rate: compression rate
    :return: gradients which lager than threshold
    """
    # start = time.time()
    current_index = 0
    u_kt.mul_(momentum)
    for param in net.parameters():
        numel = param.data.numel()
        layer_u_kt = u_kt[current_index:current_index + numel]
        if weight_decay != 0:
            param.grad.data.add_(weight_decay, param.data)
        layer_u_kt.add_(param.grad.data.view(-1).mul(lr))
        k = int(numel * rate) if int(numel * rate) != 0 else 1
        k = numel - k
        abs_layer_u_kt = layer_u_kt.abs()
        threshold = torch.kthvalue(abs_layer_u_kt, k).values
        mask = abs_layer_u_kt.gt(threshold).float()
        # print(mask.sum()-len(layer_u_kt))
        payload[current_index:current_index + numel].copy_(layer_u_kt.mul(mask))
        layer_u_kt.add_(layer_u_kt.mul(1 - mask).mul(1 / momentum - 1))
        # print(layer_u_kt.sum())
        current_index += numel
    # end = time.time()
    return payload


def DGC(net, payload, u_kt, v_kt, rate=0.01, lr=0.1, momentum=None, weight_decay=None):
    """
    :param momentum:
    :param lr:
    :param v_kt:
    :param payload:
    :param u_kt:
    :param net: model
    :param rate: compression rate
    :return: gradients which lager than threshold
    """
    start = time.time()
    current_index = 0
    u_kt.mul_(momentum)
    sum = 0
    for param in net.parameters():
        numel = param.data.numel()
        layer_u_kt = u_kt[current_index:current_index + numel]
        layer_v_kt = v_kt[current_index:current_index + numel]
        if weight_decay != 0:
            param.grad.data.add_(weight_decay, param.data)
        layer_u_kt.add_(param.grad.data.view(-1))
        layer_v_kt.add_(layer_u_kt)
        k = int(numel * rate) if int(numel * rate) != 0 else 1
        topn = [[1.0]]
        try:
            topn = torch.topk(abs(layer_v_kt), k)
        except Exception as e:
            print(e)
            print(k, layer_v_kt.nelement())
            # print(layer_v_kt)
        threshold = float(topn[0][-1])
        mask = (abs(layer_v_kt) > threshold).float()
        payload[current_index:current_index + numel].copy_(layer_v_kt.mul(mask).mul(lr))
        layer_v_kt.mul_(1 - mask)
        layer_u_kt.mul_(1 - mask)
        current_index += numel
    return payload


def Aji(net, payload, u_kt, v_kt, rate=0.01, lr=0.1, momentum=None, weight_decay=0):
    """
    :param momentum:
    :param lr:
    :param v_kt:
    :param payload:
    :param u_kt:
    :param net: model
    :param rate: compression rate
    :return: gradients which lager than threshold
    """
    start = time.time()
    current_index = 0
    # u_kt.mul_(momentum)
    for param in net.parameters():
        numel = param.data.numel()
        # layer_u_kt = u_kt[current_index:current_index + numel]
        layer_v_kt = v_kt[current_index:current_index + numel]
        # layer_u_kt.add_(param.grad.data.view(-1))
        if weight_decay != 0:
            param.grad.data.add_(weight_decay, param.data)
        layer_v_kt.add_(param.grad.data.view(-1).mul(lr))
        k = int(numel * rate) if int(numel * rate) != 0 else 1
        topn = [[1.0]]
        try:
            topn = torch.topk(abs(layer_v_kt), k)
        except Exception as e:
            print(e)
            print(k, layer_v_kt.nelement())
        threshold = float(topn[0][-1])
        mask = (abs(layer_v_kt) > threshold).float()
        payload[current_index:current_index + numel].copy_(layer_v_kt.mul(mask))
        layer_v_kt.mul_(1 - mask)
        # layer_u_kt.mul_(1 - mask)
        current_index += numel
    return payload



def server_gradient_filter(size_list, gradients, rate=0.01):
    # print('gradients', gradients)
    current_index = 0
    for size in size_list:
        numel = size
        temp = gradients[current_index:current_index + numel]
        current_index += numel
        k = int(numel * rate) if int(numel * rate) != 0 else 1
        k = numel - k
        abs_temp = temp.abs()
        threshold = torch.kthvalue(abs_temp, k).values
        mask = abs_temp.gt(threshold)
        temp.mul_(mask)
    return gradients


def ravel_sparse_gradient(temp_param):
    indices = temp_param.nonzero()
    values = temp_param[indices]
    sparse_gradient = torch.cat((indices.double(), values.double())).view(-1)
    return sparse_gradient


def unravel_sparse_gradient(sparse_gradient):
    # len is 2472266 11173962 2400w
    split = int(len(sparse_gradient) / 2)
    i = sparse_gradient[:split]
    v = sparse_gradient[split:]
    size = torch.Size([constant.MODEL_SIZE])
    # print('3',v.sum())
    try:
        dense_gradient = torch.sparse_coo_tensor(i.reshape(1, -1).long(), v.float(), size, device=torch.device('cuda'))
    except Exception as e:
        print(i, v)
        print('sum indice', sum(i))
        raise (e)
        print(size, constant.MODEL_SIZE, i[-5:], v[-5:])
        dense_gradient = torch.FloatTensor(size).zero_()

    # print(dense_gradient.sum())
    return dense_gradient
