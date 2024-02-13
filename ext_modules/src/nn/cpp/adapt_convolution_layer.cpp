#include "nn/layers/adapt_convolution_layer.h"
#include "nn/cpp/adapt_im2col.h"

#include <iostream>

namespace adapt_layer {
namespace nn {
namespace cpu {

// Bias addition occurs on the Python layer definition    
torch::Tensor AdaptConvForward(torch::Tensor input, torch::Tensor weight,
                               int kernel_h, int kernel_w,
                               int stride_h, int stride_w, int pad_h,
                               int pad_w, int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
// const int64_t nAdaptes      = input.size(2);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight  = ((inputHeight + 2 * pad_h - kernel_h -
                                 (kernel_h - 1) * (dilation_h - 1)) /
                                stride_h) +
                               1;
  const int64_t outputWidth = ((inputWidth + 2 * pad_w - kernel_w -
                                (kernel_w - 1) * (dilation_w - 1)) /
                               stride_w) +
                              1;
  const int64_t batchSize = input.size(0);

  // Initialize output and temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth},
      input.options());
  torch::Tensor columns = torch::zeros({kernel_w * kernel_h * nInputPlanes,
                                        outputHeight * outputWidth},
                                       input.options());

  // For each elt in batch, do:
  const int64_t num_kernels = nInputPlanes * columns.size(1);
  for (int b = 0; b < batchSize; b++) {
    if (input.dtype() == torch::kDouble) {
      // im2col
      AdaptIm2Col2D<double>(
          num_kernels, input[b], inputHeight, inputWidth,
          outputHeight, outputWidth, columns.size(1), kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    } else if (input.dtype() == torch::kFloat) {
      // im2col
      AdaptIm2Col2D<float>(
          num_kernels, input[b], inputHeight, inputWidth,
          outputHeight, outputWidth, columns.size(1), kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    }
    // Use PyTorch for the rest
    // Compute the convolution output
    output[b] = weight.view({weight.size(0), -1})
                    .mm(columns)
                    .view({weight.size(0), output.size(2), output.size(3),
                           output.size(4)});

    // Use PyTorch to add the bias
    //output[b] += bias.view({output[b].size(0), 1, 1, 1});
  }

  return output;
}

torch::Tensor AdaptConvBackwardInput(torch::Tensor grad_output,
                                     torch::Tensor weight, int inputHeight,
                                     int inputWidth, int kernel_h,
                                     int kernel_w, int stride_h, int stride_w,
                                     int pad_h, int pad_w, int dilation_h,
                                     int dilation_w) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
  //const int64_t nAdaptes      = grad_output.size(2);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad = torch::zeros(
      {batchSize, nInputPlanes, inputHeight, inputWidth},
      grad_output.options());

  // For each elt in batch, do:
  const int64_t num_kernels =
      nInputPlanes * inputHeight * inputWidth;
  for (int b = 0; b < batchSize; b++) {
    torch::Tensor columns = weight.view({weight.size(0), -1})
                                .transpose(1, 0)
                                .mm(grad_output[b].view({nOutputPlanes, -1}));

    if (grad_output.dtype() == torch::kDouble) {
      AdaptCol2Im2D<double>(num_kernels, columns, inputHeight,
                            inputWidth, outputHeight, outputWidth, kernel_h,
                            kernel_w, pad_h, pad_w, stride_h, stride_w,
                            dilation_h, dilation_w, input_grad[b]);
    } else if (grad_output.dtype() == torch::kFloat) {
      AdaptCol2Im2D<float>(num_kernels, columns, inputHeight,
                           inputWidth, outputHeight, outputWidth, kernel_h,
                           kernel_w, pad_h, pad_w, stride_h, stride_w,
                           dilation_h, dilation_w, input_grad[b]);
    }
  }

  return input_grad;
}

torch::Tensor AdaptConvBackwardWeight(torch::Tensor grad_output,
                                      torch::Tensor input, int kernel_h,
                                      int kernel_w, int stride_h, int stride_w,
                                      int pad_h, int pad_w, int dilation_h,
                                      int dilation_w) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t nInputPlanes  = input.size(1);
  //const int64_t nAdaptes      = grad_output.size(2);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t inputHeight   = input.size(3);
  const int64_t inputWidth    = input.size(4);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor weight_grad =
      torch::zeros({nOutputPlanes, nInputPlanes, kernel_h, kernel_w},
                   grad_output.options());
  torch::Tensor columns = torch::zeros({kernel_w * kernel_h * nInputPlanes,
                                         outputHeight * outputWidth},
                                       grad_output.options());

  // For each elt in batch, do:
  const int64_t num_kernels = nInputPlanes * columns.size(1);
  for (int64_t b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      // CUDA im2col
      AdaptIm2Col2D<double>(
          num_kernels, input[b], inputHeight, inputWidth,
          outputHeight, outputWidth, columns.size(1), kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
      AdaptIm2Col2D<float>(
          num_kernels, input[b], inputHeight, inputWidth,
          outputHeight, outputWidth, columns.size(1), kernel_h, kernel_w,
          pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    }

    // Use PyTorch for the rest
    // Compute the convolution output
    weight_grad +=
        grad_output[b]
            .view({nOutputPlanes, -1})
            .mm(columns.transpose(1, 0))
            .view({nOutputPlanes, nInputPlanes, kernel_h, kernel_w});
  }

  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace adapt_layer
