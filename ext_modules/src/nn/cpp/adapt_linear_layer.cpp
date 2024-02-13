#include "nn/layers/adapt_linear_layer.h"
#include "nn/cpp/adapt_im2col.h"

#include <iostream>

namespace adapt_layer {
namespace nn {
namespace cpu {
    
// Bias addition occurs on the Python layer definition 
    torch::Tensor AdaptLinearForward(torch::Tensor input, torch::Tensor weight) {

        auto output = input.matmul(weight.t());
        return output;
    }

    torch::Tensor AdaptLinearBackwardInput(
        torch::autograd::AutogradContext* ctx, 
        torch::Tensor grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_input = grad_output.matmul(weight);
        return grad_input;
    }

    torch::Tensor AdaptLinearBackwardWeight(
        torch::autograd::AutogradContext* ctx, 
        torch::Tensor grad_output) 
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto grad_weight = grad_output.t().matmul(input);
        return grad_weight;
    }

}  // namespace cpu
}  // namespace nn
}  // namespace adapt_layer
