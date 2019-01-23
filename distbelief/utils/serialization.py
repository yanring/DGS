import torch


def ravel_model_params(model, grads=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
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


def gradient_filter(net):
    rate = 0.01
    # paralist = []
    for param in net.parameters():
        temp = param.grad.data.clone()
        topn = torch.topk(abs(temp.view(1, -1)), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        threshold = float(topn[0][0][-1])
        temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0


def ravel_sparse_gradient(net):
    gradient_filter(net)
    temp_param = ravel_model_params(net, grads=True)
    index = torch.LongTensor(torch.where(temp_param != 0))
    value = temp_param[temp_param != 0]
    size = index.numel()
    return size, index, value
