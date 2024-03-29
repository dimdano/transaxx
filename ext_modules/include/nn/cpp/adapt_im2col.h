#ifndef PATCH_IM2COL_H_
#define PATCH_IM2COL_H_

#include <omp.h>
#include <torch/extension.h>

#include "nn/common/adapt_im2col.h"

namespace adapt_layer {
namespace nn {
namespace cpu {

template <typename T>
void AdaptIm2Col2D(const int64_t num_kernels, torch::Tensor data_im,
                   const int64_t height_im,
                   const int64_t width_im, const int64_t height_out,
                   const int64_t width_out, const int64_t width_col,
                   const int64_t kernel_h, const int64_t kernel_w,
                   const int64_t pad_h, const int64_t pad_w,
                   const int64_t stride_h, const int64_t stride_w,
                   const int64_t dilation_h, const int64_t dilation_w,
                   torch::Tensor data_col) {
  T *data_col_ptr      = data_col.data_ptr<T>();
  const T *data_im_ptr = data_im.data_ptr<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, data_im_ptr) private(index) \
    schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::AdaptIm2Col2D(index, data_im_ptr, height_im, width_im,
                          height_out, width_out, width_col, kernel_h, kernel_w,
                          pad_h, pad_w, stride_h, stride_w, dilation_h,
                          dilation_w, data_col_ptr);
  }
}

template <typename T>
void AdaptCol2Im2D(const int64_t num_kernels, torch::Tensor data_col,
                   const int64_t height,
                   const int64_t width, const int64_t output_height,
                   const int64_t output_width, const int64_t kernel_h,
                   const int64_t kernel_w, const int64_t pad_h,
                   const int64_t pad_w, const int64_t stride_h,
                   const int64_t stride_w, const int64_t dilation_h,
                   const int64_t dilation_w, torch::Tensor data_im) {
  const T *data_col_ptr = data_col.data_ptr<T>();
  T *data_im_ptr        = data_im.data_ptr<T>();
  int64_t index;
#pragma omp parallel for shared(data_col_ptr, data_im_ptr) private(index) \
    schedule(static)
  for (index = 0; index < num_kernels; index++) {
    common::AdaptCol2Im2D(index, data_col_ptr, height, width,
                          output_height, output_width, kernel_h, kernel_w,
                          pad_h, pad_w, stride_h, stride_w, dilation_h,
                          dilation_w, data_im_ptr);
  }
}

}  // namespace cpu
}  // namespace nn
}  // namespace adapt_layer
#endif
