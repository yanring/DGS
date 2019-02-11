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


# def gradient_filter(net):
#     rate = 0.5
#     # paralist = []
#     for param in net.parameters():
#         temp = param.grad.data.clone()
#         topn = torch.topk(abs(temp.view(1, -1)), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
#         threshold = float(topn[0][0][-1])
#         # temp[abs(temp) >= threshold] = 0
#         param.grad.data[abs(param.grad.data) < threshold] = 0
def gradient_filter(net):
    rate = 0.5
    threshold = 0.0001
    # paralist = []
    for param in net.parameters():
        # temp = param.grad.data.clone()
        # topn = torch.topk(abs(temp.view(1, -1)), int(temp.nelement() * rate) if int(temp.nelement() * rate) != 0 else 1)
        # threshold = float(topn[0][0][-1])
        # temp[abs(temp) >= threshold] = 0
        param.grad.data[abs(param.grad.data) < threshold] = 0


def ravel_sparse_gradient(net, lr=1):
    # gradient_filter(net)
    temp_param = ravel_model_params(net, grads=True).mul_(lr)
    # threshold = 0.0001
    threshold = 0.000002 * abs(temp_param).sum()
    print(abs(temp_param).sum())
    temp_param[abs(temp_param) < threshold] = 0
    indices = temp_param.nonzero()
    values = temp_param[indices]
    # value = temp_param[temp_param != 0]
    # print(values.sum())
    # size = indices.numel()
    sparse_gradient = torch.cat((indices.float(), values)).view(-1)
    # print(indices.t().size(),values.view(-1).size(),temp_param.size())
    return sparse_gradient


def unravel_sparse_gradient(sparse_gradient):
    # len is 2472266
    split = int(len(sparse_gradient) / 2)
    i = sparse_gradient[:split]
    v = sparse_gradient[split:]
    # print(i.t().long().size(), v.size(),torch.Size([2472266]))
    dense_gradient = torch.sparse.FloatTensor(i.reshape(1, -1).long(), v, torch.Size([2472266])).to_dense()
    # print(dense_gradient.sum())
    return dense_gradient
