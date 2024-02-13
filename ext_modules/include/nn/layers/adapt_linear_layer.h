#ifndef ADAPT_LINEAR_LAYER_H_
#define ADAPT_LINEAR_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace adapt_layer {
namespace nn {

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// FORWARD DECLARATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {

    torch::Tensor AdaptLinearForward(torch::Tensor input, torch::Tensor weight);

    torch::Tensor AdaptLinearBackwardInput(torch::autograd::AutogradContext* ctx, torch::Tensor grad_output);
    
    torch::Tensor AdaptLinearBackwardWeight(torch::autograd::AutogradContext* ctx, torch::Tensor grad_output);
    
}  // namespace cuda
#endif

namespace cpu {
    
    torch::Tensor AdaptLinearForward(torch::Tensor input, torch::Tensor weight);

    torch::Tensor AdaptLinearBackwardInput(torch::autograd::AutogradContext* ctx, torch::Tensor grad_output);
    
    torch::Tensor AdaptLinearBackwardWeight(torch::autograd::AutogradContext* ctx, torch::Tensor grad_output);
        
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor AdaptLinearForward(torch::Tensor input, torch::Tensor weight) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.device().is_cuda()) {
    CHECK_CUDA(weight);

    return cuda::AdaptLinearForward(input, weight);
      
  } else
#endif
  {
    CHECK_CPU(weight);
      
    return cpu::AdaptLinearForward(input, weight);
  }
}

    
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> AdaptLinearBackward(
        torch::autograd::AutogradContext* ctx, 
        torch::Tensor grad_output) 
    {
    
    #ifndef __NO_CUDA__  // CUDA compilation only
      if (grad_output.device().is_cuda()) {

        auto grad_input = cuda::AdaptLinearBackwardInput(ctx, grad_output);
        auto grad_weight = cuda::AdaptLinearBackwardWeight(ctx, grad_output);          
        auto saved = ctx->get_saved_variables();
        auto bias = saved[2];
        torch::Tensor grad_bias;
        if (bias.defined()) {
            grad_bias = grad_output.sum(0);
        }
        return std::make_tuple(grad_input, grad_weight, grad_bias);
          
      } else
    #endif
      {

        auto grad_input = cpu::AdaptLinearBackwardInput(ctx, grad_output);
        auto grad_weight = cpu::AdaptLinearBackwardWeight(ctx, grad_output);
        auto saved = ctx->get_saved_variables();
        auto bias = saved[2];
        torch::Tensor grad_bias;
        if (bias.defined()) {
            grad_bias = grad_output.sum(0);
        }
        return std::make_tuple(grad_input, grad_weight, grad_bias);
      }
    
    }  
  

}  // namespace nn
}  // namespace adapt_linear

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adapt_linear_forward", &adapt_layer::nn::AdaptLinearForward,
        "Forward adapt linear");
  m.def("adapt_linear_backward", &adapt_layer::nn::AdaptLinearBackward,
        "Backward adapt linear");
}

#endif
