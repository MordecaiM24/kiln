#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <tuple>

namespace {

constexpr int64_t kMaxFusedN = 65536;
constexpr int kThreads = 256;
constexpr int kRowsPerDwGroup = 32;
constexpr int kMaxDwGroups = 512;

__inline__ __device__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float block_sum(float value) {
  __shared__ float warp_sums[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  return value;
}

template <typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    float* __restrict__ rstd,
    int64_t n,
    float eps) {
  const int64_t row = blockIdx.x;
  const int64_t row_offset = row * n;
  float square_sum0 = 0.0f;
  float square_sum1 = 0.0f;
  const int64_t stride = blockDim.x * 2LL;
  for (int64_t col = threadIdx.x; col < n; col += stride) {
    const float value0 = static_cast<float>(x[row_offset + col]);
    square_sum0 += value0 * value0;
    const int64_t col1 = col + blockDim.x;
    if (col1 < n) {
      const float value1 = static_cast<float>(x[row_offset + col1]);
      square_sum1 += value1 * value1;
    }
  }
  float square_sum = square_sum0 + square_sum1;
  square_sum = block_sum(square_sum);
  __shared__ float row_rstd;
  if (threadIdx.x == 0) {
    row_rstd = 1.0f / sqrtf(square_sum / static_cast<float>(n) + eps);
    rstd[row] = row_rstd;
  }
  __syncthreads();
  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    const float value = static_cast<float>(x[row_offset + col]);
    const float scale = static_cast<float>(weight[col]);
    y[row_offset + col] = static_cast<scalar_t>(value * row_rstd * scale);
  }
}

__global__ void rmsnorm_forward_half2_kernel(
    const __half2* __restrict__ x,
    const __half2* __restrict__ weight,
    __half2* __restrict__ y,
    float* __restrict__ rstd,
    int64_t n2,
    float eps) {
  const int64_t row = blockIdx.x;
  const int64_t row_offset = row * n2;
  float square_sum0 = 0.0f;
  float square_sum1 = 0.0f;
  const int64_t stride = blockDim.x * 2LL;
  for (int64_t col = threadIdx.x; col < n2; col += stride) {
    const float2 value0 = __half22float2(x[row_offset + col]);
    square_sum0 += value0.x * value0.x + value0.y * value0.y;
    const int64_t col1 = col + blockDim.x;
    if (col1 < n2) {
      const float2 value1 = __half22float2(x[row_offset + col1]);
      square_sum1 += value1.x * value1.x + value1.y * value1.y;
    }
  }
  float square_sum = block_sum(square_sum0 + square_sum1);
  __shared__ float row_rstd;
  if (threadIdx.x == 0) {
    row_rstd = 1.0f / sqrtf(square_sum / static_cast<float>(n2 * 2) + eps);
    rstd[row] = row_rstd;
  }
  __syncthreads();
  for (int64_t col = threadIdx.x; col < n2; col += blockDim.x) {
    const float2 value = __half22float2(x[row_offset + col]);
    const float2 scale = __half22float2(weight[col]);
    y[row_offset + col] = __floats2half2_rn(
        value.x * row_rstd * scale.x, value.y * row_rstd * scale.y);
  }
}

template <typename scalar_t>
__global__ void rmsnorm_dx_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const float* __restrict__ rstd,
    scalar_t* __restrict__ grad_x,
    int64_t n) {
  const int64_t row = blockIdx.x;
  const int64_t row_offset = row * n;
  const float inv_rms = rstd[row];
  float dot = 0.0f;
  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    const float x_hat = __fmul_rn(static_cast<float>(x[row_offset + col]), inv_rms);
    const float wdy = __fmul_rn(static_cast<float>(weight[col]),
                                static_cast<float>(grad_y[row_offset + col]));
    dot = __fadd_rn(dot, __fmul_rn(x_hat, wdy));
  }
  dot = block_sum(dot);
  __shared__ float correction;
  if (threadIdx.x == 0) {
    correction = dot / static_cast<float>(n);
  }
  __syncthreads();
  for (int64_t col = threadIdx.x; col < n; col += blockDim.x) {
    const float x_hat = __fmul_rn(static_cast<float>(x[row_offset + col]), inv_rms);
    const float wdy = __fmul_rn(static_cast<float>(weight[col]),
                                static_cast<float>(grad_y[row_offset + col]));
    const float centered = __fsub_rn(wdy, __fmul_rn(x_hat, correction));
    grad_x[row_offset + col] = static_cast<scalar_t>(__fmul_rn(inv_rms, centered));
  }
}

template <typename scalar_t>
__global__ void rmsnorm_dw_partial_kernel(
    const scalar_t* __restrict__ grad_y,
    const scalar_t* __restrict__ x,
    const float* __restrict__ rstd,
    float* __restrict__ partial,
    int64_t m,
    int64_t n,
    int64_t rows_per_group) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t group = blockIdx.y;
  if (col >= n) {
    return;
  }
  const int64_t row_begin = group * rows_per_group;
  const int64_t row_end = min(m, row_begin + rows_per_group);
  float sum = 0.0f;
  for (int64_t row = row_begin; row < row_end; ++row) {
    const int64_t offset = row * n + col;
    const float x_hat = __fmul_rn(static_cast<float>(x[offset]), rstd[row]);
    sum = __fadd_rn(sum, __fmul_rn(static_cast<float>(grad_y[offset]), x_hat));
  }
  partial[group * n + col] = sum;
}

void check_common(const torch::Tensor& x, const torch::Tensor& weight) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "internal input must be 2D");
  TORCH_CHECK(x.is_contiguous(), "input rows must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  const int64_t n = x.size(1);
  TORCH_CHECK(n > 0, "normalized dimension must be non-empty");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == n,
              "weight shape must be (", n, ")");
  TORCH_CHECK(n <= kMaxFusedN, "N=", n, " exceeds MAX_FUSED_N=", kMaxFusedN);
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "input and weight must have the same dtype");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16 ||
              x.scalar_type() == torch::kBFloat16 ||
              x.scalar_type() == torch::kFloat32,
              "supported dtypes are float16, bfloat16, and float32");
  TORCH_CHECK(x.device() == weight.device(), "input and weight must be on the same device");
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_cuda(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    double eps) {
  check_common(x, weight);
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t m = x.size(0);
  const int64_t n = x.size(1);
  auto y = torch::empty_like(x);
  auto rstd = torch::empty({m}, x.options().dtype(torch::kFloat32));
  if (m == 0) {
    return {y, rstd};
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool aligned_half2 = x.scalar_type() == torch::kFloat16 && (n % 2 == 0) &&
      (reinterpret_cast<uintptr_t>(x.const_data_ptr()) % alignof(__half2) == 0) &&
      (reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % alignof(__half2) == 0) &&
      (reinterpret_cast<uintptr_t>(y.mutable_data_ptr()) % alignof(__half2) == 0);
  if (aligned_half2) {
    rmsnorm_forward_half2_kernel<<<m, kThreads, 0, stream>>>(
        reinterpret_cast<const __half2*>(x.const_data_ptr()),
        reinterpret_cast<const __half2*>(weight.const_data_ptr()),
        reinterpret_cast<__half2*>(y.mutable_data_ptr()), rstd.mutable_data_ptr<float>(),
        n / 2, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, rstd};
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "rmsnorm_cuda_forward", [&] {
        rmsnorm_forward_kernel<scalar_t><<<m, kThreads, 0, stream>>>(
            x.const_data_ptr<scalar_t>(), weight.const_data_ptr<scalar_t>(),
            y.mutable_data_ptr<scalar_t>(), rstd.mutable_data_ptr<float>(), n,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, rstd};
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_cuda_backward(
    const torch::Tensor& grad_y,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& rstd) {
  check_common(x, weight);
  TORCH_CHECK(grad_y.is_cuda() && grad_y.is_contiguous(),
              "grad_y must be a contiguous CUDA tensor");
  TORCH_CHECK(grad_y.sizes() == x.sizes() && grad_y.scalar_type() == x.scalar_type(),
              "grad_y must match input shape and dtype");
  TORCH_CHECK(rstd.is_cuda() && rstd.is_contiguous() &&
              rstd.scalar_type() == torch::kFloat32 && rstd.dim() == 1 &&
              rstd.size(0) == x.size(0),
              "rstd must be contiguous fp32 with shape (M,)");
  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t m = x.size(0);
  const int64_t n = x.size(1);
  auto grad_x = torch::empty_like(x);
  const int64_t groups = std::max<int64_t>(1, std::min<int64_t>(
      kMaxDwGroups, (m + kRowsPerDwGroup - 1) / kRowsPerDwGroup));
  const int64_t rows_per_group = m == 0 ? 0 : (m + groups - 1) / groups;
  auto dw_partial = torch::zeros({groups, n}, x.options().dtype(torch::kFloat32));
  if (m == 0) {
    return {grad_x, dw_partial};
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 dw_grid((n + kThreads - 1) / kThreads, groups);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "rmsnorm_cuda_backward", [&] {
        rmsnorm_dx_kernel<scalar_t><<<m, kThreads, 0, stream>>>(
            grad_y.const_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
            weight.const_data_ptr<scalar_t>(), rstd.const_data_ptr<float>(),
            grad_x.mutable_data_ptr<scalar_t>(), n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        rmsnorm_dw_partial_kernel<scalar_t><<<dw_grid, kThreads, 0, stream>>>(
            grad_y.const_data_ptr<scalar_t>(), x.const_data_ptr<scalar_t>(),
            rstd.const_data_ptr<float>(), dw_partial.mutable_data_ptr<float>(),
            m, n, rows_per_group);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, dw_partial};
}

}  // namespace

TORCH_LIBRARY(kiln, m) {
  m.def("rmsnorm_cuda(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
  m.def("rmsnorm_cuda_backward(Tensor grad_y, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(kiln, CUDA, m) {
  m.impl("rmsnorm_cuda", &rmsnorm_cuda);
  m.impl("rmsnorm_cuda_backward", &rmsnorm_cuda_backward);
}
