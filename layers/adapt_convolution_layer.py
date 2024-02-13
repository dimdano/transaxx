# +
import torch
import torch.nn as nn
import os.path as osp
import os
#for quantization
import torch.utils.data
import pytorch_quantization.utils
import pytorch_quantization.nn.modules._utils as _utils
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn
#

#from .torch_utils import _ConvNd, _size_2_t, Union, Tensor, Optional, _pair
import torch.nn.functional as F 
import torch.nn as nn
from torch.nn import Parameter
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import math
# -

import math

#import _adaptconv_ext._adapt_convolution as adapt_conv
from .layer_utils import _pair


##### JIT compilation definitions #####
from torch.utils.cpp_extension import load
compute_arch = 'compute_70'
base_dir = os.environ.get('PYTHONPATH').split(os.pathsep)[-1]
prefix = base_dir + '/ext_modules/src/nn/cpp'
prefix_cuda = base_dir + '/ext_modules/src/nn/cuda'
include_dir = base_dir + '/ext_modules/include'
source_basename = 'adapt_convolution_layer'
###################################################

class AdaptConv2DFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size, stride, padding,
                dilation, groups,
               fake_quant, quantizer, quantizer_w, amax, amax_w, quant_limit, axx_conv2d_kernel):
        '''
        self.save_for_backward(input, weight, bias, torch.tensor(kernel_size),
                               torch.tensor(stride), torch.tensor(padding),
                               torch.tensor(dilation), torch.tensor(groups))
        '''
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
            
        quant_weight = quantizer_w(weight)
        quant_input = quantizer(input)   
                        
        if (amax == None or fake_quant):    
            return F.conv2d(quant_input, quant_weight, bias, stride, padding, dilation, groups)
            
        else:
            
            quant_input = quant_input.to(dtype=torch.int8)
            quant_weight = quant_weight.to(dtype=torch.int8)
             
            #slow temporary version of grouped conv2d using simple conv2d (split+concat)
            #support only for group = input_dim = outputdim (i.e. mobilenetv2)
            if groups > 1: 
                out = torch.empty(0)
                for i in range(0, groups):
                    filters = quant_weight[i:(i+1)]                   
                    o =  axx_conv2d_kernel.adapt_conv_forward(quant_input[:, i:(i+1)], filters, kernel_size[0], kernel_size[1],
                                                         stride[0], stride[1], padding[0],
                                                         padding[1], dilation[0], dilation[1]) 
                    out = torch.cat((out, o), dim=1)

                out = out / ((quant_limit / amax) * ((quant_limit / amax_w)))

            else: 
                out = axx_conv2d_kernel.adapt_conv_forward(quant_input, quant_weight, kernel_size[0], kernel_size[1],
                                                     stride[0], stride[1], padding[0],
                                                     padding[1], dilation[0], dilation[1])
                out = out / ((quant_limit / amax) * ((quant_limit / amax_w)))
                
            if bias is not None:
                out = out + bias.reshape(1, quant_weight.shape[0], 1, 1)

            return out
    

    @staticmethod
    def backward(ctx, grad_output):
        '''
        input, weight, bias, kernel_size, stride, padding, dilation, groups = self.saved_tensors
        grad_input, grad_weight, grad_bias = adapt_conv.adapt_conv_backward(
            grad_output, input, weight, bias, kernel_size[0], kernel_size[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0],
            dilation[1])
        '''

        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None, None


class AdaptConv2D(nn.Module):

    def __init__(
            self,
            in_channels,  # Input channels to convolution
            out_channels,  # Output channels from convolution
            kernel_size=1,  # Filter size
            stride=1,  # Stride
            padding=0,  # Padding
            dilation=1,  # Dilation
            groups=1,   # Groups
            bias=True,
            axx_mult='mul8s_acc', #name of axx_mult header file
            quant_bits = 8, #quantization bits of axx_mult
            fake_quant = False): #works only for 8bit

        super(AdaptConv2D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        self.axx_mult = axx_mult
        self.quant_bits = quant_bits
        self.fake_quant = fake_quant   
        self.flops = 0  
        self.params = 0  
        self.flops_percentage = 0
        self.power_percentage = 0
        self.params_percentage = 0
        
     

        ######### Parameter initialization ######### 
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size[0],
                         self.kernel_size[1]))

        #we need to deal with bias transfering from default model
        if isinstance(bias, bool) and not bias:
            # Bias is explicitly set to False, do not include bias
            self.register_parameter('bias', None)
        elif isinstance(bias, torch.Tensor) and bias is not None:
            # Bias is already a tensor, leave it as it is
            self.bias = nn.Parameter(bias)
        elif isinstance(bias, bool) and bias:
            # Bias is a boolean set to True, initialize it
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            # Handle other cases (bias is None, or unexpected type)
            self.register_parameter('bias', None)
 
        self.reset_parameters()
        
        ######### Quantization #########
        unsigned=False
        if unsigned: 
            self.quant_limit = pow(2,self.quant_bits)-1
        else:
            self.quant_limit = pow(2,self.quant_bits-1)-1

        self.quant_desc = QuantDescriptor(num_bits=self.quant_bits, fake_quant=self.fake_quant, unsigned=unsigned, calib_method='histogram')
        self.quantizer = TensorQuantizer(self.quant_desc)
        self.quantizer_w = TensorQuantizer(self.quant_desc)
        
    def set_axx_kernel(self):
        
        #Jit compilation method for cpp/cuda extention
        quantization_flags = ' -DAXX_MULT=' + self.axx_mult + ' -DQUANT_BITS=' + str(self.quant_bits)
        self.axx_conv2d_kernel = None if self.fake_quant else load(name='PyInit_conv2d_'+ self.axx_mult , sources=[
                                 osp.join(prefix, source_basename + '.cpp'),
                                 osp.join(prefix_cuda, source_basename + '.cu')],
                                 extra_include_paths=[include_dir, base_dir], extra_cflags=[quantization_flags, ' -fopenmp -O3'],
                                 extra_cuda_cflags=[quantization_flags, ' --gpu-architecture=' + compute_arch],
                                 verbose=False)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def flops_power_mem_percent(self, total_flops, total_params, axx_power):
        self.flops_percentage = self.flops*100/total_flops
        self.power_percentage = self.flops_percentage * axx_power # 1000x times more to avoid a lot of decimals 
        

    def forward(self, x):
        return AdaptConv2DFunction.apply(x, self.weight, self.bias,
                                              self.kernel_size, self.stride,
                                              self.padding, self.dilation, self.groups, self.fake_quant, self.quantizer, self.quantizer_w, self.quantizer.amax, self.quantizer_w.amax, self.quant_limit, self.axx_conv2d_kernel)
