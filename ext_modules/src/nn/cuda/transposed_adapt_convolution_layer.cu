#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "nn/cuda/adapt_im2col.cuh"

namespace adapt_layer {
namespace nn {
namespace cuda {

torch::Tensor TransposedAdaptConvForward(torch::Tensor input,
                                         torch::Tensor weight,
                                         torch::Tensor bias, int kH, int kW,
                                         int dH, int dW, int padH, int padW,
                                         int dilationH, int dilationW) {
  // Useful dimensions to have
  const int64_t nInputPlanes  = weight.size(0);
  const int64_t nOutputPlanes = weight.size(1);
  //const int64_t nAdaptes      = input.size(2);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  const int64_t outputHeight =
      dH * (inputHeight - 1) + kH + (kH - 1) * (dilationH - 1) - 2 * padH;
  const int64_t outputWidth =
      dW * (inputWidth - 1) + kW + (kW - 1) * (dilationW - 1) - 2 * padW;
  const int64_t batchSize = input.size(0);
  const int64_t inputBatchStride =
      nInputPlanes * inputHeight * inputWidth;

  // Initialize output and temporary columns
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth},
      input.options());
  torch::Tensor columns = torch::zeros(
      {kW * kH * nOutputPlanes, inputHeight * inputWidth},
      input.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Get cuda stream
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t n = weight.size(1) * weight.size(2) * weight.size(3);
    const int64_t k = weight.size(0);

    if (input.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
      //            input.data<double>() + b * inputBatchStride, m,
      //            weight.data<double>(), n, &beta, columns.data<double>(), m);
    } else if (input.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
       //           input.data<float>() + b * inputBatchStride, m,
       //           weight.data<float>(), n, &beta, columns.data<float>(), m);
    } else {
      printf("Can only support double and float\n");
      std::exit(-1);
    }
    CUDA_CHECK(cudaGetLastError())

    AdaptCol2Im2DLauncher(columns, nOutputPlanes, outputHeight,
                          outputWidth, inputHeight, inputWidth, kH, kW, padH,
                          padW, dH, dW, dilationH, dilationW, output[b]);

    // Use PyTorch to add the bias
    output[b] += bias.view({output[b].size(0), 1, 1, 1});
  }

  return output;
}

torch::Tensor TransposedAdaptConvBackwardInput(torch::Tensor grad_output,
                                               torch::Tensor weight,
                                               int inputHeight, int inputWidth,
                                               int kH, int kW, int dH, int dW,
                                               int padH, int padW,
                                               int dilationH, int dilationW) {
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
  torch::Tensor columns = torch::zeros(
      {kW * kH * nOutputPlanes, inputHeight * inputWidth},
      grad_output.options());

  // For each elt in batch, do:
  const int64_t inputBatchStride =
      nInputPlanes * inputHeight * inputWidth;
  for (int b = 0; b < batchSize; b++) {
    AdaptIm2Col2DLauncher(grad_output[b], nOutputPlanes,
                          outputHeight, outputWidth, inputHeight, inputWidth,
                          columns.size(1), kH, kW, padH, padW, dH, dW,
                          dilationH, dilationW, columns);

    // Get cuda stream
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t k = weight.size(1) * weight.size(2) * weight.size(3);
    const int64_t n = weight.size(0);

    if (grad_output.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      //            columns.data<double>(), m, weight.data<double>(), k, &beta,
      //            input_grad.data<double>() + b * inputBatchStride, m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      //            columns.data<float>(), m, weight.data<float>(), k, &beta,
      //            input_grad.data<float>() + b * inputBatchStride, m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return input_grad;
}

torch::Tensor TransposedAdaptConvBackwardWeight(torch::Tensor grad_output,
                                                torch::Tensor input, int kH,
                                                int kW, int dH, int dW,
                                                int padH, int padW,
                                                int dilationH, int dilationW) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = grad_output.size(1);
  const int64_t nInputPlanes  = input.size(1);
  //const int64_t nAdaptes      = grad_output.size(2);
  const int64_t outputHeight  = grad_output.size(2);
  const int64_t outputWidth   = grad_output.size(3);
  const int64_t inputHeight   = input.size(3);
  const int64_t inputWidth    = input.size(4);
  const int64_t batchSize     = grad_output.size(0);
  const int64_t inputBatchStride =
      nInputPlanes * inputHeight * inputWidth;

  // Initialize output and temporary columns
  torch::Tensor weight_grad = torch::zeros(
      {nInputPlanes, nOutputPlanes, kH, kW}, grad_output.options());
  torch::Tensor columns = torch::zeros(
      {kW * kH * nOutputPlanes, inputHeight * inputWidth},
      grad_output.options());

  // For each elt in batch, do:
  for (int b = 0; b < batchSize; b++) {
    // Create the column matrix from the grad output as we would for the input
    // in the standard conv_forward
    AdaptIm2Col2DLauncher(grad_output[b], nOutputPlanes,
                          outputHeight, outputWidth, inputHeight, inputWidth,
                          columns.size(1), kH, kW, padH, padW, dH, dW,
                          dilationH, dilationW, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Propagate the gradients from the outputs to the weights using GEMM
    // Note that GEMM expects column major matrices
    const int64_t m =
        weight_grad.size(1) * weight_grad.size(2) * weight_grad.size(3);
    const int64_t n = weight_grad.size(0);
    const int64_t k = columns.size(1);

    if (grad_output.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 1.0;
      cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), k,
                  input.data<double>() + b * inputBatchStride, k, &beta,
                  weight_grad.data<double>(), m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 1.0;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), k,
                  input.data<float>() + b * inputBatchStride, k, &beta,
                  weight_grad.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return weight_grad;
}

}  // namespace cuda
}  // namespace nn
}  // namespace adapt_layer
