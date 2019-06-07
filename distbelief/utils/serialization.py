import time
import torch

from distbelief.utils import constant

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


def gradient_filter(param):
    rate = 0.01
    grad = param
    # temp = param.grad.data

    # param = param.grad.data
    topn = torch.kthvalue(abs(grad.view(1, -1)), int(grad.nelement() * (1 - rate)))
    threshold = float(topn[0])
    # topn = torch.topk(abs(temp.view(1, -1)), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
    # threshold = float(topn[0][0][-1])
    param[abs(param) < threshold] = 0
    # print(abs(param.grad.data).sum())
    # print(threshold)
    return threshold


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
    start = time.time()
    current_index = 0
    u_kt.mul_(momentum)
    for param in net.parameters():
        numel = param.data.numel()
        layer_u_kt = u_kt[current_index:current_index + numel]
        if weight_decay != 0:
            param.grad.data.add_(weight_decay, param.data)
        layer_u_kt.add_(param.grad.data.view(-1))
        k = int(numel * rate) if int(numel * rate) != 0 else 1
        topn = [[1.0]]
        try:
            topn = torch.topk(abs(layer_u_kt), k)
        except Exception as e:
            print(e)
            print(k, layer_u_kt.nelement())
            return payload
        threshold = float(topn[0][-1])
        mask = (abs(layer_u_kt) > threshold).float()
        payload[current_index:current_index + numel].copy_(layer_u_kt.mul(mask).mul(lr))
        layer_u_kt.copy_(layer_u_kt.mul(1 - mask).mul(1 / momentum))
        current_index += numel
    end = time.time()
    return payload


def DGC(net, payload, u_kt, v_kt, rate=0.01, lr=0.1, momentum=None):
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
        payload[current_index:current_index + numel].copy_(layer_v_kt.mul(mask).mul_(lr))
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
    sum = 0
    for param in net.parameters():
        numel = param.data.numel()
        # layer_u_kt = u_kt[current_index:current_index + numel]
        if weight_decay != 0:
            param.grad.data.add_(weight_decay, param.data)
        layer_v_kt = v_kt[current_index:current_index + numel]
        # layer_u_kt.add_(param.grad.data.view(-1))
        layer_v_kt.add_(param.grad.data.view(-1))
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
        payload[current_index:current_index + numel].copy_(layer_v_kt.mul(mask).mul_(lr))
        layer_v_kt.mul_(1 - mask)
        # layer_u_kt.mul_(1 - mask)
        current_index += numel
    return payload

def worker_gradient_filter(net, rate=0.01):
    start = time.time()
    # rate = 0.01
    paralist = []
    # threshold = 0.0001
    # paralist = []
    for param in net.parameters():
        temp = param.grad.data.clone()
        topn = torch.topk(abs(temp.view(1, -1)), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        threshold = float(topn[0][0][-1])
        if threshold == 0:
            temp.zero_()
            paralist.append(temp)
            continue
        temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0
        paralist.append(temp)
    end = time.time()
    return paralist
    # print(end - start)


def server_gradient_filter(size_list, gradients, rate=0.01):
    # print('gradients', gradients)
    current_index = 0
    for size in size_list:
        numel = size
        temp = gradients[current_index:current_index + numel]
        current_index += numel
        topn = torch.topk(abs(temp), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        threshold = float(topn[0][-1])
        # topn = torch.kthvalue(abs(temp), int(size * (1 - rate)))
        # threshold = float(topn[0])
        if threshold > 0:
            temp[abs(temp) < threshold] = 0
    return gradients


def unravel_model_grad(model, parameter_update):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    current_index = 0  # keep track of where to read from parameter_update
    for parameter in model.parameters():
        numel = parameter.grad.data.numel()
        size = parameter.grad.data.size()
        parameter.grad.data.copy_(parameter_update[current_index:current_index + numel].view(size))
        current_index += numel


def ravel_sparse_gradient(temp_param):
    # gradient_filter(net)
    # temp_param = ravel_model_params(net, grads=True).mul_(lr)
    # threshold = 0.0001
    # threshold = 0.000001 * abs(temp_param).sum()
    # print(abs(temp_param).sum())
    # temp_param[abs(temp_param) < threshold] = 0
    indices = temp_param.nonzero()
    values = temp_param[indices]
    # if len(indices) > 3000000:
    #     print("why???", len(indices), values.sum())
    #     return torch.FloatTensor([1, 0.1])
    # value = temp_param[temp_param != 0]
    # print(values.sum())
    # size = indices.numel()
    sparse_gradient = torch.cat((indices.float(), values)).view(-1)
    # print(indices.t().size(),values.view(-1).size(),temp_param.size())
    return sparse_gradient


def unravel_sparse_gradient(sparse_gradient):
    # len is 2472266 11173962 2400w
    split = int(len(sparse_gradient) / 2)
    i = sparse_gradient[:split]
    v = sparse_gradient[split:]
    size = torch.Size([constant.MODEL_SIZE])
    try:
        dense_gradient = torch.sparse.FloatTensor(i.reshape(1, -1).long(), v, size).to_dense()
    except Exception as e:
        print(e)
        print(size, constant.MODEL_SIZE, i[:20])
        dense_gradient = torch.FloatTensor(size).zero_()

    # print(dense_gradient.sum())
    return dense_gradient
