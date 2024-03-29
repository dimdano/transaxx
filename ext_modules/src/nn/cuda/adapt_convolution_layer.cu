#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "nn/cuda/adapt_im2col.cuh"

#include <iostream>

namespace adapt_layer {
namespace nn {
namespace cuda {

// Bias addition occurs on the Python layer definition
torch::Tensor AdaptConvForward(torch::Tensor input, torch::Tensor weight,
                               int kH, int kW, int dH, int dW,
                               int padH, int padW, int dilationH, int dilationW) {
  // Useful dimensions to have
  const int64_t nOutputPlanes = weight.size(0);
  const int64_t nInputPlanes  = weight.size(1);
  //const int64_t nAdaptes      = input.size(2);
  const int64_t inputHeight   = input.size(2);
  const int64_t inputWidth    = input.size(3);
  
  const int64_t outputHeight =
      ((inputHeight + 2 * padH - kH - (kH - 1) * (dilationH - 1)) / dH) + 1;
  const int64_t outputWidth =
      ((inputWidth + 2 * padW - kW - (kW - 1) * (dilationW - 1)) / dW) + 1;
  const int64_t batchSize = input.size(0);
/*
std::cout << weight.size(0) << "\n";
std::cout << weight.size(1) << "\n";
std::cout << weight.size(2) << "\n";
std::cout << weight.size(3)<< "\n";

std::cout << input.size(0) << "\n";
std::cout << input.size(1) << "\n";
std::cout << input.size(2) << "\n";
std::cout << input.size(3) << "\n";

std::cout << outputHeight << "\n";
std::cout << outputWidth << "\n";
std::cout << input.sizes() << "\n";

std::cout << "------------\n";
  
 */
  // Initialize output and temporary columns
  const int64_t extra_pad = ((outputHeight * outputWidth + 64 - 1)/64)*64 - (outputHeight * outputWidth);
  const int64_t padded_dim1 = ((outputHeight * outputWidth + 64 - 1)/64)*64;

  const int64_t padded_dim2 = ((nOutputPlanes + 64 - 1)/64)*64;

  const int64_t padded_dim3 = ((kW * kH * nInputPlanes + 64 - 1)/64)*64;


  //auto out_options1 = input.options;
  //auto out_options = out_options1.dtype(torch::kInt32);
  
  torch::Tensor output = torch::zeros(
      {batchSize, nOutputPlanes, outputHeight, outputWidth},
      input.options().dtype(torch::kInt32));
  torch::Tensor columns = torch::zeros(
      {kW * kH * nInputPlanes, outputHeight * outputWidth},
      input.options().dtype(input.dtype()));

  // For case of cublas sgemm:
  //const int64_t outputBatchStride =
  //    nOutputPlanes * outputHeight * outputWidth;
  
  //for case of custom sgemm
const int64_t outputBatchStride =
      nOutputPlanes*outputHeight * outputWidth; //nOutputPlanes * outputHeight * outputWidth;

//std::cout << "M: " << weight.size(0) << " N: " << columns.size(1) << " K: " <<  weight.size(1) * weight.size(2) * weight.size(3) << "\n";

  for (int b = 0; b < batchSize; b++) {
    
    cudaError_t cuda_err;

//at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    // CUDA im2col
    AdaptIm2Col2DLauncher(input[b], nInputPlanes, inputHeight,
                          inputWidth, outputHeight, outputWidth,
                          columns.size(1), kH, kW, padH, padW, dH, dW,
                          dilationH, dilationW, columns);

    // Get cuda stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    //cudaStream_t stream = at::cuda::getCurrentCUDAStream();
   


   //at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
   //at::cuda::setCurrentCUDAStream(defaultStream);

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t k = weight.size(1) * weight.size(2) * weight.size(3);
    const int64_t n = weight.size(0);
    
//std::cout << m << "\n";
//std::cout << n << "\n";
//std::cout << weight.size(0) << "\n";
//std::cout << weight.size(1) << "\n";
//std::cout << weight.size(2) << "\n";
//std::cout << weight.size(3) << "\n";
//std::cout << k << "\n";

//std::cout << columns.size(0) << "\n";
//std::cout << columns.size(1) << "\n";
//std::cout << outputBatchStride << "\n";
//std::cout << output.size(0) << "\n";


/*
    if (input.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<double>(), m, weight.data<double>(), k, &beta,
                  output.data<double>() + b * outputBatchStride, m);
    } else if (input.dtype() == torch::kFloat) {
*/ 
    const float alpha = 1.0;
      const float beta  = 0.0;
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        //          columns.data<float>(), m, weight.data<float>(), k, &beta,
        //          output.data<float>() + b * outputBatchStride, m);
      
    
//cuda_err = ReferenceGemm_Launcher(m, n, k, alpha, columns.data<float>(), m, weight.data<float>(), k, beta, output.data<float>() + b * outputBatchStride, m);
//need to manually change data_ptr types if kernel changes
cuda_err = ReferenceGemm_Launcher(n, m, k, alpha, weight.data_ptr<int8_t>(), k, columns.data_ptr<int8_t>(), m, beta, output.data_ptr<int>() + b * outputBatchStride, m);



  //  }
    
    //CUDA_CHECK(cudaGetLastError())

    // Use PyTorch to add the bias
	//if(has_bias)	
    //  output[b] += bias.view({output[b].size(0),1, 1});
  }
	return output;
  //return (output.narrow(2, 0, outputHeight * outputWidth)).narrow(1,0,nOutputPlanes);
}

torch::Tensor AdaptConvBackwardInput(torch::Tensor grad_output,
                                     torch::Tensor weight, int inputHeight,
                                     int inputWidth, int kH, int kW, int dH,
                                     int dW, int padH, int padW, int dilationH,
                                     int dilationW) {
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
  torch::Tensor columns = torch::zeros(
      {kW * kH * nInputPlanes, outputHeight * outputWidth},
      grad_output.options());

  // For each elt in batch, do:
  const int64_t outputBatchStride =
      nOutputPlanes * outputHeight * outputWidth;
  for (int b = 0; b < batchSize; b++) {
    // Get cuda stream
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

    // Call the GEMM function (note that it expects column major matrices)
    const int64_t m = columns.size(1);
    const int64_t n = weight.size(1) * weight.size(2) * weight.size(3);
    const int64_t k = weight.size(0);

    if (grad_output.dtype() == torch::kDouble) {
      const double alpha = 1.0;
      const double beta  = 0.0;
      //cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
      //            grad_output.data<double>() + b * outputBatchStride, m,
      //            weight.data<double>(), n, &beta, columns.data<double>(), m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 0.0;
      //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
      //            grad_output.data<float>() + b * outputBatchStride, m,
      //            weight.data<float>(), n, &beta, columns.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())

    AdaptCol2Im2DLauncher(columns, nInputPlanes, inputHeight,
                          inputWidth, outputHeight, outputWidth, kH, kW, padH,
                          padW, dH, dW, dilationH, dilationW, input_grad[b]);
  }

  return input_grad;
}

torch::Tensor AdaptConvBackwardWeight(torch::Tensor grad_output,
                                      torch::Tensor input, int kH, int kW,
                                      int dH, int dW, int padH, int padW,
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

  // Initialize output and temporary columns
  torch::Tensor weight_grad = torch::zeros(
      {nOutputPlanes, nInputPlanes, kH, kW}, grad_output.options());
  torch::Tensor columns = torch::zeros(
      {kW * kH * nInputPlanes, outputHeight * outputWidth},
      grad_output.options());

  // For each elt in batch, do:
  const int64_t outputBatchStride =
      nOutputPlanes * outputHeight * outputWidth;
  for (int b = 0; b < batchSize; b++) {
    // CUDA im2col
    AdaptIm2Col2DLauncher(input[b], nInputPlanes, inputHeight,
                          inputWidth, outputHeight, outputWidth,
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
                  grad_output.data<double>() + b * outputBatchStride, k, &beta,
                  weight_grad.data<double>(), m);
    } else if (grad_output.dtype() == torch::kFloat) {
      const float alpha = 1.0;
      const float beta  = 1.0;
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                  columns.data<float>(), k,
                  grad_output.data<float>() + b * outputBatchStride, k, &beta,
                  weight_grad.data<float>(), m);
    }
    CUDA_CHECK(cudaGetLastError())
  }

  return weight_grad;
}

}  // namespace cuda
}  // namespace nn
}  // namespace adapt_layer

