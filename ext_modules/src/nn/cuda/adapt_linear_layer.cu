#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAUtils.h>

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
torch::Tensor AdaptLinearForward(torch::Tensor input, torch::Tensor weight) {

    auto m = input.size(0);
    auto n = weight.size(1);
    auto k = input.size(1);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t cuda_err;

    auto result = torch::empty({m, n}, input.options().dtype(torch::kInt32));

    float alpha = 1.0f;
    float beta = 0.0f;
    
    
    int lda = k, ldb = n, ldc = n;


    cuda_err = ReferenceGemm_Launcher(m, n, k, alpha, input.data_ptr<int8_t>(), lda, weight.data_ptr<int8_t>(), ldb, beta, result.data_ptr<int>(), ldc);
    


/*
    
    auto result = torch::empty({m, n}, input.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    cublasHandle_t handle;
    cublasCreate(&handle);


    cublasSgemm(handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              n,
              m,
              k,
              &alpha,
              weight.data_ptr<float>(),
              k,
              input.data_ptr<float>(),
              k,
              &beta,
              result.data_ptr<float>(),
              n);

    cublasDestroy(handle);
*/
    AT_CUDA_CHECK(cuda_err);
    
    return result;

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
    
 
}  // namespace cuda
}  // namespace nn
}  // namespace adapt_layer

