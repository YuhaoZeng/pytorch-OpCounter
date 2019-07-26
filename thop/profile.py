import logging

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .count_hooks import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

register_hooks = {
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose2d: count_convtranspose2d,

    nn.BatchNorm1d: count_bn,
    nn.BatchNorm2d: count_bn,
    nn.BatchNorm3d: count_bn,

    nn.ReLU: count_relu,
    nn.ReLU6: count_relu,
    nn.LeakyReLU: count_relu,

    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,

    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: None,
}


def profile(model, inputs, custom_ops=None, verbose=True):
    handler_collection = []
    if custom_ops is None:
        custom_ops = {}

        
    ops_list=[]
    memory_list=[]
    layer_type_list=[]
    layer_count = []
 


    def add_hooks(m):
        if len(list(m.children())) > 0:
            return

        if hasattr(m, "total_ops") or hasattr(m, "total_params") or hasattr(m,"total_memory"):
            logger.warning("Either .total_ops or .total_params or .total_memory is already defined in %s." 
                           "Be careful, it might change your code's behavior." % str(m))

        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        m.register_buffer('total_memory', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        m_type = type(m)
        fn = None
        if m_type in custom_ops:  # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        if fn is None:
            if verbose:
                print("THOP has not implemented counting method for ", m)
        else:
            if verbose:
                print("Register FLOP counter for module %s" % str(m))
            handler = m.register_forward_hook(fn)
            handler_collection.append(handler)
        
        layer_count = 0
        ops_list.append(m.total_ops.item())
        memory_list.append(m.total_params.item())
        layer_type_list.append(str(m))
        print('{0:8}{1:15}{2:15}{3:15}{4:15}'.format(layer_count,str(m),m.total_params.item(),m.total_memory.item(),m.total_ops.item()))


    # original_device = model.parameters().__next__().device
    training = model.training   
    
    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    total_ops = 0
    total_params = 0
    total_memory = 0

    

    
    
    print('{0:^8}{1:^15}{2:^15}{3:^15}{4:^15}'.format("number","layer_type","params","memory","flops"))
    for m in model.modules():
        if len(list(m.children())) > 0:  # skip for non-leaf module
            continue
        layer_count = layer_count + 1
        total_ops += m.total_ops
        total_params += m.total_params
        total_memory += m.total_memory
        layer_out = int(m.out_features) if hasattr(m,'out_features') else 0
        
        #ops_list.append(m.total_ops.item())
        #memory_list.append(m.total_params.item())
        #layer_type_list.append(type(m))
        
        #print('The No.%d layer:' % layer_count)
        #print('The type of this layer:%s, The parameters of this layer:%d, The output number of this layer:%d' % (type(m),int(m.total_params),layer_out))
        #print('The memory of this layer:%d' % int(m.total_memory))
        #print('{0:8}{1:20}{2:15}{3:15}{4:15}'.format(layer_count,m.total_params.item(),m.total_memory.item(),m.total_ops.item(),"100"))

        
    total_ops = total_ops.item()
    total_params = total_params.item()
    total_memory = total_memory.item()

    # reset model to original status
    model.train(training)
    for handler in handler_collection:
        handler.remove()
    
    #total_ops = clever_format(total_ops)
    #total_params = clever_format(total_params)
    print('The Total Flops:',total_ops)
    print('The Total parameters:',total_params)
    print('The Total Memory:',total_memory)
    return total_ops, total_params,ops_list,memory_list,layer_type_list
