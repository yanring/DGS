import time

import torch

from distbelief.utils import messaging

current_model_size = None


def ravel_model_params(model, grads=False, cuda=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    if cuda or messaging.isCUDA:
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


#
# def mp_gradient_filter(net):
#     import torch.multiprocessing as mp
#     # TODO 用numpy做多进程
#     start = time.time()
#     # res = list(map(gradient_filter, net.parameters()))
#     pool = mp.Pool(processes=4)
#     a = [i.grad for i in net.parameters()]
#     pool.map(gradient_filter, a)
#     # # for i in net.parameters():
#     #     # pool.apply(gradient_filter,(i.grad.data,))
#     pool.close()
#     pool.join()
#     # print('ddddddddddddddddddddone')
#     # for param,threshold in zip(net.parameters(),res):
#     #     print(abs(param.grad.data).sum())
#     # param.grad.data = torch.where(param<threshold,param,torch.full_like(param, 0))
#     # param.grad.data[abs(param.grad.data) < threshold] = 0
#     end = time.time()
#     # print('time:',end - start)


def worker_gradient_executor(net, payload, rate=0.01, lr=0.1):
    '''
    :param net: model
    :param rate: compression rate
    :return: gradients which lager than threshold
    '''
    start = time.time()
    current_index = 0
    m_parameter = payload
    for param in net.parameters():
        numel = param.data.numel()
        m_parameter[current_index:current_index + numel] = param.grad.data.clone().view(-1)
        temp = m_parameter[current_index:current_index + numel]
        topn = torch.topk(abs(temp), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        threshold = float(topn[0][-1])
        temp[abs(temp) <= threshold] = 0
        param.grad.data[abs(param.grad.data) > threshold] = 0
        current_index += numel
        # paralist.append(temp)
    end = time.time()
    return m_parameter.mul_(lr)


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
        temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0
        paralist.append(temp)
    end = time.time()
    return paralist
    # print(end - start)


def server_gradient_filter(net, gradients, rate=0.01):
    start = time.time()
    current_index = 0
    for param in net.parameters():
        numel = param.data.numel()
        temp = gradients[current_index:current_index + numel]
        topn = torch.topk(abs(temp), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        threshold = float(topn[0][-1])
        temp[abs(temp) < threshold] = 0
    end = time.time()


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
    # value = temp_param[temp_param != 0]
    # print(values.sum())
    # size = indices.numel()
    sparse_gradient = torch.cat((indices.float(), values)).view(-1)
    # print(indices.t().size(),values.view(-1).size(),temp_param.size())
    return sparse_gradient


def unravel_sparse_gradient(sparse_gradient):
    # len is 2472266 11173962
    split = int(len(sparse_gradient) / 2)
    i = sparse_gradient[:split]
    v = sparse_gradient[split:]
    # print(i.t().long().size(), v.size(),torch.Size([2472266]))
    dense_gradient = torch.sparse.FloatTensor(i.reshape(1, -1).long(), v, torch.Size([2472266])).to_dense()
    # print(dense_gradient.sum())
    return dense_gradient
