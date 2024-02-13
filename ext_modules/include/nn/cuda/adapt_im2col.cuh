#ifndef PATCH_IM2COL_CUH_
#define PATCH_IM2COL_CUH_

#include <torch/extension.h>

#include "cuda_helper.h"
#include "nn/common/adapt_im2col.h"

#define FLOAT float
#define INT int
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

#define TILE_DIM 16

#define STR(s) STR2(s)
#define STR2(s) #s
#define EXPAND(s) s

#include STR(axx_mults/EXPAND(AXX_MULT).h)
#define QUANT_MASK (1 << QUANT_BITS) - 1

namespace adapt_layer {
namespace nn {
namespace cuda {

__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

const int TILE_SIZE = 16;
const int VECTOR_SIZE = 4;


template<typename T1, typename T2, typename T3>
__global__ void axx_gemm_default(size_t m, size_t n, size_t k,
                                             const  T1 *a, size_t lda, const T2 *b, size_t ldb,
                                             T3 *c, size_t ldc)
{
    T3 value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ T1 As[TILE_DIM][TILE_DIM];
    __shared__ T2 Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

         if (i*TILE_DIM + threadIdx.x < k && Row < m)
             As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = T1(0);

         if (i*TILE_DIM + threadIdx.y < k && Col < n)
             Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = T2(0);

         __syncthreads();

         for (int t = 0; t < TILE_DIM; ++t){
            value += __ldg(&lut[uint8_t(As[threadIdx.y][t])&QUANT_MASK][uint8_t(Bs[t][threadIdx.x])&QUANT_MASK]);
         //value += As[threadIdx.y][t] * Bs[t][threadIdx.x];
        }
         __syncthreads();
    }

    if (Row < m && Col < n)
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) +
           (blockIdx.x * blockDim.x) + threadIdx.x] = value;
}


/// GEMM launcher.
template<typename T1, typename T2, typename T3>
cudaError_t ReferenceGemm_Launcher(
  int M,
  int N,
  int K,
  float alpha,
  T1 const *A,
  int lda,
  T2 const *B,
  int ldb,
  float beta,
  T3 *C,
  int ldc) {


  dim3 blockSize(16, 16, 1);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y, 1);
  axx_gemm_default<T1,T2,T3><<<gridSize, blockSize>>>(M,N,K,A,lda,B,ldb,C,ldc);

  return cudaGetLastError();
}
    
    
template<typename T1, typename T2, typename T3>    
__global__ void matrix_multiply_shared(const T1* A, const T2* B, T3* C, int m, int n, int k, int lda, int ldb, int ldc) {
     
     __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];
  
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
  
    float Cvalue = 0;
  
    for (int i = 0; i < (k-1)/TILE_DIM+1; i++) {
        if (row < m && i*TILE_DIM+tx < k) {
            A_shared[ty][tx] = A[row*lda + i*TILE_DIM+tx];
        } else {
            A_shared[ty][tx] = 0.0;
        }

        if (col < n && i*TILE_DIM+ty < k) {
            B_shared[ty][tx] = B[(i*TILE_DIM+ty)*ldb + col];
        } else {
            B_shared[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int j = 0; j < TILE_DIM; j++) {
            //Cvalue += A_shared[ty][j] * B_shared[j][tx];
            Cvalue += __ldg(&lut[uint8_t(A_shared[ty][j])&QUANT_MASK][uint8_t(B_shared[j][tx])&QUANT_MASK]);
        }

        __syncthreads();
    }
  
    if (row < m && col < n) {
        C[row*ldc+col] = Cvalue;
    }
}



/// GEMM launcher.
template<typename T1, typename T2, typename T3>
cudaError_t ReferenceGemm_Launcher_new(
  int M,
  int N,
  int K,
  float alpha,
  T1 const *A,
  int lda,
  T2 const *B,
  int ldb,
  float beta,
  T3 *C,
  int ldc) {


    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N - 1) / TILE_DIM + 1, (M - 1) / TILE_DIM + 1);

    matrix_multiply_shared<T1,T2,T3><<<numBlocks, threadsPerBlock>>>(A, B, C, M, N, K, lda, ldb, ldc);
        
    
    return cudaGetLastError();
}
    
    



	
template <typename T>
__global__ void AdaptIm2Col2DKernel(
    const int64_t num_kernels, const T *data_im_ptr,
    const int64_t height_im, const int64_t width_im, const int64_t height_out,
    const int64_t width_out, const int64_t width_col, const int64_t kernel_h,
    const int64_t kernel_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t dilation_h,
    const int64_t dilation_w, T *data_col_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }

  common::AdaptIm2Col2D(index, data_im_ptr, height_im, width_im,
                        height_out, width_out, width_col, kernel_h, kernel_w,
                        pad_h, pad_w, stride_h, stride_w, dilation_h,
                        dilation_w, data_col_ptr);
}

void AdaptIm2Col2DLauncher(torch::Tensor data_im, const int64_t channels,
                           const int64_t height_im,
                           const int64_t width_im, const int64_t height_out,
                           const int64_t width_out, const int64_t width_col,
                           const int64_t kernel_h, const int64_t kernel_w,
                           const int64_t pad_h, const int64_t pad_w,
                           const int64_t stride_h, const int64_t stride_w,
                           const int64_t dilation_h, const int64_t dilation_w,
                           torch::Tensor data_col) {
  const int64_t num_kernels = channels * width_col;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  // Launch channels * width_col kernels, with each kernel responsible for
  // copying a the convolutions over a single channel.
  AT_DISPATCH_ALL_TYPES(
      data_col.scalar_type(), "AdaptIm2Col2DKernel", ([&] {
        AdaptIm2Col2DKernel<<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_im.data_ptr<scalar_t>(), height_im,
            width_im, height_out, width_out, width_col, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            data_col.data_ptr<scalar_t>());
        CUDA_CHECK(cudaGetLastError())
      }));
}

template <typename T>
__global__ void AdaptCol2Im2DKernel(
    const int64_t num_kernels, const T *data_col_ptr,
    const int64_t height, const int64_t width, const int64_t output_height,
    const int64_t output_width, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w, const int64_t stride_h,
    const int64_t stride_w, const int64_t dilation_h, const int64_t dilation_w,
    T *data_im_ptr) {
  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_kernels) { return; }
  common::AdaptCol2Im2D(index, data_col_ptr,  height, width,
                        output_height, output_width, kernel_h, kernel_w, pad_h,
                        pad_w, stride_h, stride_w, dilation_h, dilation_w,
                        data_im_ptr);
}

void AdaptCol2Im2DLauncher(torch::Tensor data_col, const int64_t channels,
                           const int64_t height,
                           const int64_t width, const int64_t output_height,
                           const int64_t output_width, const int64_t kernel_h,
                           const int64_t kernel_w, const int64_t pad_h,
                           const int64_t pad_w, const int64_t stride_h,
                           const int64_t stride_w, const int64_t dilation_h,
                           const int64_t dilation_w, torch::Tensor data_im) {
  const int64_t num_kernels = channels * height * width;
  const dim3 blocks((num_kernels + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);

  AT_DISPATCH_FLOATING_TYPES(
      data_col.scalar_type(), "AdaptCol2Im2DKernel", ([&] {
        AdaptCol2Im2DKernel<scalar_t><<<blocks, CUDA_NUM_THREADS>>>(
            num_kernels, data_col.data<scalar_t>(), height, width,
            output_height, output_width, kernel_h, kernel_w, pad_h, pad_w,
            stride_h, stride_w, dilation_h, dilation_w,
            (data_im.data<scalar_t>()));
      }));
  CUDA_CHECK(cudaGetLastError())

}

}  // namespace cuda
}  // namespace nn
}  // namespace adapt_layer
#endif
