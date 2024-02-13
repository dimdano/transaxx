#ifndef ADAPT_CONVOLUTION_LAYER_H_
#define ADAPT_CONVOLUTION_LAYER_H_

#include <torch/extension.h>
#include <vector>

#include "cuda_helper.h"

namespace adapt_layer {
namespace nn {

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// FUNCTIONAL LAYER DECLARATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

#ifndef __NO_CUDA__  // CUDA compilation only
namespace cuda {
torch::Tensor AdaptConvForward(torch::Tensor input, torch::Tensor weight,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w, int pad_h,
                               int pad_w, int dilation_h, int dilation_w);

torch::Tensor AdaptConvBackwardInput(torch::Tensor grad_output,
                                     torch::Tensor weight, int inputHeight,
                                     int inputWidth, int kernel_h,
                                     int kernel_w, int stride_h, int stride_w,
                                     int pad_h, int pad_w, int dilation_h,
                                     int dilation_w);

torch::Tensor AdaptConvBackwardWeight(torch::Tensor grad_output,
                                      torch::Tensor input, int kernel_h,
                                      int kernel_w, int stride_h, int stride_w,
                                      int pad_h, int pad_w, int dilation_h,
                                      int dilation_w);
}  // namespace cuda
#endif

namespace cpu {
torch::Tensor AdaptConvForward(torch::Tensor input, torch::Tensor weight,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w, int pad_h,
                               int pad_w, int dilation_h, int dilation_w);

torch::Tensor AdaptConvBackwardInput(torch::Tensor grad_output,
                                     torch::Tensor weight, int inputHeight,
                                     int inputWidth, int kernel_h,
                                     int kernel_w, int stride_h, int stride_w,
                                     int pad_h, int pad_w, int dilation_h,
                                     int dilation_w);

torch::Tensor AdaptConvBackwardWeight(torch::Tensor grad_output,
                                      torch::Tensor input, int kernel_h,
                                      int kernel_w, int stride_h, int stride_w,
                                      int pad_h, int pad_w, int dilation_h,
                                      int dilation_w);
}  // namespace cpu

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// CALL FUNCTION IMPLEMENTATIONS
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

torch::Tensor AdaptConvForward(torch::Tensor input, torch::Tensor weight,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w, int pad_h,
                               int pad_w, int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (input.device().is_cuda()) {
    CHECK_CUDA(weight);

    return cuda::AdaptConvForward(input, weight, kernel_h, kernel_w,
                                  stride_h, stride_w, pad_h, pad_w, dilation_h,
                                  dilation_w);
  } else
#endif
  {
    CHECK_CPU(weight);
    return cpu::AdaptConvForward(input, weight, kernel_h, kernel_w,
                                 stride_h, stride_w, pad_h, pad_w, dilation_h,
                                 dilation_w);
  }
}

std::vector<torch::Tensor> AdaptConvBackward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w) {
  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(weight);

#ifndef __NO_CUDA__  // CUDA compilation only
  if (grad_output.device().is_cuda()) {
    CHECK_CUDA(input);
    CHECK_CUDA(weight);

    torch::Tensor grad_input = cuda::AdaptConvBackwardInput(
        grad_output, weight, input.size(3), input.size(4), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    torch::Tensor grad_weight = cuda::AdaptConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3, 4});

    return {grad_input, grad_weight, grad_bias};
  } else
#endif
  {
    CHECK_CPU(input);
    CHECK_CPU(weight);

    torch::Tensor grad_input = cpu::AdaptConvBackwardInput(
        grad_output, weight, input.size(3), input.size(4), kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);

    torch::Tensor grad_weight = cpu::AdaptConvBackwardWeight(
        grad_output, input, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w);

    torch::Tensor grad_bias = grad_output.sum({0, 2, 3, 4});

    return {grad_input, grad_weight, grad_bias};
  }
}

}  // namespace nn
}  // namespace adapt_layer

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adapt_conv_forward", &adapt_layer::nn::AdaptConvForward,
        "Forward adapt convolution");
  m.def("adapt_conv_backward", &adapt_layer::nn::AdaptConvBackward,
        "Backward adapt convolution");
}

#endif
