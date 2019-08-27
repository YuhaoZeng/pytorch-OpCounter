import argparse

import torch
import torch.nn as nn

multiply_adds = 1

def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def count_convNd(m, x, y):
    x = x[0]
    cin = m.in_channels
    stride = m.in_channels
    # batch_size = x.size(0)

    kernel_ops = m.weight.size()[2:].numel()
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])
    
    
    #memory
    #batch_size = x.size()[0]
    #batch_size=1
    #cin = x.size()[1]
    #cout,out_h,out_w = y.size()[1:]
    #mread = batch_size * (x.size()[1:].numel() + num_params(m))
    #mwrite = batch_size * cout * out_h * out_w
    #mread = x.element() + num_params
    mread = num_params(m)
    mwrite = output_elements
    
    m.total_memory = torch.Tensor([int(mread+mwrite)])

#
#def count_conv2d(m, x, y):
#    x = x[0]

#    cin = m.in_channels
#    cout = m.out_channels
#    kh, kw = m.kernel_size
#    batch_size = x.size()[0]

#    out_h = y.size(2)
#    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
#    kernel_ops = multiply_adds * kh * kw
#    bias_ops = 1 if m.bias is not None else 0
#    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
#    output_elements = batch_size * out_w * out_h * cout
#    total_ops = output_elements * ops_per_element * cin // m.groups

#    m.total_ops = torch.Tensor([int(total_ops)])


def count_convtranspose2d(m, x, y):
    x = x[0]

    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    # batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw * cin // m.groups
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    # output_elements = batch_size * out_w * out_h * cout
    # ops_per_element = m.weight.nelement()

    output_elements = y.nelement()
    total_ops = output_elements * ops_per_element

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #memory
    #batch_size = x.size()[0]
    #batch_size = 1
    #cin = x.size()[1]
    #cout,out_h,out_w = y.size()[1:]
    #mread = batch_size * (x.size()[1:].numel() + num_patams(m))
    #mwrite = batch_size * cout * out_h * out_w
    
    #mread = x.nelement() + num_params(m)
    mread = num_params(m)
    mwrite = output_elements
    m.total_memory = torch.Tensor([int(mread+mwrite)])


def count_bn(m, x, y):
    x = x[0]

    nelements = x.numel()
    # subtract, divide, gamma, beta
    total_ops = 4 * nelements

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #batch_size, in_c, in_h, in_w = x.size()

    #mread = batch_size * (x.size()[1:].numel() + 2 * in_c)
    #mwrite = x.size().numel()
    
    #mread = nelements
    mwrite = nelements
    
    m.total_memory = torch.Tensor([int(mwrite)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #
    #batch_size = x.size()[0]
    #mread = x.numel()
    mwrite = y.numel()
    
    m.total_memory = torch.Tensor([int(mwrite)])


def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #
    #batch_size = x.size()[0]
    
    #mread = nfeatures
    mwrite = nfeatures
    
    m.total_memory = torch.Tensor([int(mwrite)])



def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #mread = total_add+1
    mwrite = num_elements

    m.total_memory = torch.Tensor([int(mwrite)])


def count_adap_avgpool(m, x, y):
    kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
    total_add = torch.prod(kernel) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
    
    #batch_size = x.size()[0]
    #batch_size = 1
    #mread = batch_size * x.size()[1:].numel()
    #mwrite = batch_size * y.size()[1:].numel()
    
    mread = torch.prod(kernel)
    mwrite = num_elements

    m.total_memory = torch.Tensor([int(mread+mwrite)])


def count_linear(m, x, y):
    # per output element
    bias_ops = 1 if m.bias is not None else 0
    num_elements = y.numel()
    total_ops=num_elements * (m.in_features+bias_ops)
    #total_mul = m.in_features
    #total_add = m.in_features - 1
    #num_elements = y.numel()
    #total_ops = (total_mul + total_add) * num_elements

    m.total_ops = torch.Tensor([int(total_ops)])
    
    ##
    #batch_size = x.size()[0]
    
    
    mread = num_params(m)
    mwrite = num_elements

    m.total_memory = torch.Tensor([int(mread+mwrite)])
