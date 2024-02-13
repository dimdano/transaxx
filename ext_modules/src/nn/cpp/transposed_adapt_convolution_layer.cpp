#include "nn/layers/transposed_adapt_convolution_layer.h"
#include "nn/cpp/adapt_im2col.h"

namespace adapt_conv {
namespace nn {
namespace cpu {

torch::Tensor TransposedAdaptConvForward(torch::Tensor input,
                                         torch::Tensor weight,
                                         torch::Tensor bias, int kernel_h,
                                         int kernel_w, int stride_h,
                                         int stride_w, int pad_h, int pad_w,
                                         int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(0);
  const int64_t nOutputPlanes = weight.size(1);
  //const int64_t nAdaptes      = input.size(2);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight  = stride_h * (inputHeight - 1) + kernel_h +
                               (kernel_h - 1) * (dilation_h - 1) - 2 * pad_h;
  const int64_t outputWidth = stride_w * (inputWidth - 1) + kernel_w +
                              (kernel_w - 1) * (dilation_w - 1) - 2 * pad_w;
  const int64_t batchSize = input.size(0);

  // Initialize output and temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth},
      input.options());

  // For each elt in batch, do:
  const int64_t num_kernels =
      nOutputPlanes * outputHeight * outputWidth;
  for (int b = 0; b < batchSize; b++) {
    // Use PyTorch for the initial matrix multiplication
    torch::Tensor columns = weight.view({weight.size(0), -1})
                                .transpose(1, 0)
                                .mm(input[b].view({nInputPlanes, -1}));

    if (input.dtype() == torch::kDouble) {
      AdaptCol2Im2D<double>(num_kernels, columns, outputHeight,
                            outputWidth, inputHeight, inputWidth, kernel_h,
                            kernel_w, pad_h, pad_w, stride_h, stride_w,
                            dilation_h, dilation_w, output[b]);
    } else if (input.dtype() == torch::kFloat) {
      AdaptCol2Im2D<float>(num_kernels, columns, outputHeight,
                           outputWidth, inputHeight, inputWidth, kernel_h,
                           kernel_w, pad_h, pad_w, stride_h, stride_w,
                           dilation_h, dilation_w, output[b]);
    }

    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1, 1});
  }

  return output;
}

torch::Tensor TransposedAdaptConvBackwardInput(
    torch::Tensor grad_output, torch::Tensor weight, int inputHeight,
    int inputWidth, int kernel_h, int kernel_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(0);
  const int64_t nOutputPlanes = weight.size(1);
  //const int64_t nAdaptes      = grad_output.size(2);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t batchSize     = grad_output.size(0);

  // Initialize output and temporary columns
  torch::Tensor input_grad = torch::zeros(
      {batchSize, nInputPlanes, inputHeight, inputWidth},
      grad_output.options());
  torch::Tensor columns = torch::zeros({kernel_w * kernel_h * nOutputPlanes,
                                        inputHeight * inputWidth},
                                       grad_output.options());

  // For each elt in batch, do:
  const int64_t num_kernels = nOutputPlanes * columns.size(1);
  for (int b = 0; b < batchSize; b++) {
    if (grad_output.dtype() == torch::kDouble) {
      AdaptIm2Col2D<double>(
          num_kernels, grad_output[b], outputHeight, outputWidth,
          inputHeight, inputWidth, columns.size(1), kernel_h, kernel_w, pad_h,
          pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
      AdaptIm2Col2D<float>(
          num_kernels, grad_output[b], outputHeight, outputWidth,
          inputHeight, inputWidth, columns.size(1), kernel_h, kernel_w, pad_h,
          pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    }

    // Use PyTorch for the matrix multiplication
    input_grad[b] =
        weight.view({weight.size(0), -1})
            .mm(columns)
            .view({nInputPlanes, inputHeight, inputWidth});
  }

  return input_grad;
}

torch::Tensor TransposedAdaptConvBackwardWeight(
    torch::Tensor grad_output, torch::Tensor input, int kernel_h, int kernel_w,
    int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h,
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
      torch::zeros({nInputPlanes, nOutputPlanes, kernel_h, kernel_w},
                   grad_output.options());
  torch::Tensor columns = torch::zeros({kernel_w * kernel_h * nOutputPlanes,
                                        inputHeight * inputWidth},
                                       grad_output.options());

  // For each elt in batch, do:
  const int64_t num_kernels = nOutputPlanes * columns.size(1);
  for (int b = 0; b < batchSize; b++) {
    // Create the column matrix from the grad output as we would for the input
    // in the standard conv_forward
    if (grad_output.dtype() == torch::kDouble) {
      AdaptIm2Col2D<double>(
          num_kernels, grad_output[b], outputHeight, outputWidth,
          inputHeight, inputWidth, columns.size(1), kernel_h, kernel_w, pad_h,
          pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    } else if (grad_output.dtype() == torch::kFloat) {
      AdaptIm2Col2D<float>(
          num_kernels, grad_output[b], outputHeight, outputWidth,
          inputHeight, inputWidth, columns.size(1), kernel_h, kernel_w, pad_h,
          pad_w, stride_h, stride_w, dilation_h, dilation_w, columns);
    }

    // Use PyTorch for the final matrix multiplication
    weight_grad +=
        input[b]
            .view({input[b].size(0), -1})
            .mm(columns.transpose(1, 0))
            .view({nInputPlanes, nOutputPlanes, kernel_h, kernel_w});
  }

  return weight_grad;
}

}  // namespace cpu
}  // namespace nn
}  // namespace adapt_conv
