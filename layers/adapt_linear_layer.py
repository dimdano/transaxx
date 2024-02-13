#from .torch_utils import _ConvNd, _size_2_t, Union, Tensor, Optional, _pair
import torch
from torch import Tensor
import torch.nn.functional as F 
import torch.nn as nn
from torch.nn import Parameter


#import _adaptconv_ext._adapt_convolution as adapt_conv
from .layer_utils import *


##### JIT compilation definitions #####
from torch.utils.cpp_extension import load
compute_arch = 'compute_70'
base_dir = os.environ.get('PYTHONPATH').split(os.pathsep)[-1]
prefix = base_dir + '/ext_modules/src/nn/cpp'
prefix_cuda = base_dir + '/ext_modules/src/nn/cuda'
include_dir = base_dir + '/ext_modules/include'
source_basename = 'adapt_linear_layer'
###################################################


# Important: Transposing the weight matrix in matmul makes it non contigous so we added 
# a 'contiguous' operation (which creates some memory+latency overhead) in order to keep the historical weight layout 
class AdaPT_Linear_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, has_bias, fake_quant, quantizer, quantizer_w, amax, amax_w, quant_limit, axx_linear_kernel):
        ctx.save_for_backward(input, weight, bias)
        ctx.has_bias = has_bias
        
        quant_weight = quantizer_w(weight)
        quant_input = quantizer(input)   
    
        if (amax == None or fake_quant):
            output = torch.matmul(quant_input, quant_weight.t())
            
        else:           
            quant_input = quant_input.to(dtype=torch.int8)
            quant_weight = quant_weight.to(dtype=torch.int8)
            
            #case for Vit models. workaround for 3 dimensional input using typical matrix mult
            if input.dim() > 2:
                # Reshape input to a 2D tensor with shape (batch_size * ..., in_features)
                quant_input_2d = quant_input.reshape(-1, quant_input.size(-1))

                # Compute output using matrix multiplication
                output = axx_linear_kernel.adapt_linear_forward(quant_input_2d.reshape(-1, quant_input.size(-1)), 
                                                                quant_weight.t().contiguous())

                # Reshape output to original shape of input tensor
                output_shape = quant_input.size()[:-1] + (quant_weight.size(0),)
                output = output.reshape(output_shape)
           
            else:           
                output = axx_linear_kernel.adapt_linear_forward(quant_input, quant_weight.t().contiguous())
                
            output = output/((quant_limit/amax)*(quant_limit/amax_w))                           
                                      
        if has_bias:
            return output + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        has_bias = ctx.has_bias
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            if grad_output.dim() == 2:
                grad_weight = grad_output.t().matmul(input)
            elif grad_output.dim() == 3:
                grad_weight = grad_output.transpose(1, 2).matmul(input)
        if has_bias and ctx.needs_input_grad[2]:
            if grad_output.dim() == 2:
                grad_bias = grad_output.sum(dim=0)
            elif grad_output.dim() == 3:
                grad_bias = grad_output.sum(dim=(0, 1))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


class AdaPT_Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, axx_mult='mul8s_acc', quant_bits = 8, fake_quant = False):
        
        super(AdaPT_Linear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.axx_mult = axx_mult
        self.quant_bits = quant_bits
        self.fake_quant = fake_quant   
        self.flops = 0  
        self.params = 0  
        self.flops_percentage = 0
        self.power_percentage = 0
        self.params_percentage = 0  
    
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
        
        ######### Jit compilation method for cpp/cuda extention #########
        quantization_flags = ' -DAXX_MULT=' + self.axx_mult + ' -DQUANT_BITS=' + str(self.quant_bits)
        self.axx_linear_kernel = load(name='PyInit_linear_'+ self.axx_mult, sources=[
                                 osp.join(prefix, source_basename + '.cpp'),
                                 osp.join(prefix_cuda, source_basename + '.cu')],
                                 extra_include_paths=[include_dir, base_dir], extra_cflags=[quantization_flags, ' -fopenmp -O3'],
                                 extra_cuda_cflags=[quantization_flags, ' --gpu-architecture=' + compute_arch],
                                 verbose=False)
   
    def flops_power_mem_percent(self, total_flops, total_params, axx_power):
        self.flops_percentage = self.flops*100/total_flops
        #self.params_percentage = self.params*100/total_params   #TODO: get param value
        self.power_percentage = self.flops_percentage * axx_power # 1000x times more to avoid a lot of decimals 
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):       
        x = AdaPT_Linear_Function.apply(x, self.weight, self.bias, self.has_bias, self.fake_quant, self.quantizer, self.quantizer_w, self.quantizer.amax, self.quantizer_w.amax, self.quant_limit, self.axx_linear_kernel)
        
        return x
