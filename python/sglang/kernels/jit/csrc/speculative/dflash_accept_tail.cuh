/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// The tail of a DFLASH decode step's acceptance in one launch, for request i of the batch:
//   new_seq_lens[i]                = prefix_lens[i] + int64(commit_lens[i])
//   new_seq_lens_buf[indices[i]]   = new_seq_lens[i]
//   output_tokens_buf[indices[i]]  = bonus[i]
// These are the values SGLang writes with an int32 -> int64 conversion, an int64 add and two index_put calls (the
// spec-v2 overlap relay's publish and stash). Integer copies and one integer add only. The indices of one call are
// distinct (checked by the caller), so every element is written exactly once.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang::dflash_accept_tail {

constexpr int kThreads = 128;

struct Params {
  const int64_t* prefix_lens;  // [bs]
  const int32_t* commit_lens;  // [bs]
  const int64_t* bonus;        // [bs]
  const int64_t* indices;      // [bs]
  int64_t* new_seq_lens;       // [bs]
  int64_t* new_seq_lens_buf;   // [pool]
  int64_t* output_tokens_buf;  // [pool]
  int bs;
};

template <bool kUsePDL>
__global__ __launch_bounds__(kThreads) void accept_tail_kernel(const __grid_constant__ Params p) {
  device::PDLWaitPrimary<kUsePDL>();
  for (int i = threadIdx.x; i < p.bs; i += kThreads) {
    const int64_t n = p.prefix_lens[i] + static_cast<int64_t>(p.commit_lens[i]);
    const int64_t row = p.indices[i];
    p.new_seq_lens[i] = n;
    p.new_seq_lens_buf[row] = n;
    p.output_tokens_buf[row] = p.bonus[i];
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct AcceptTail {
  static void
  run(const tvm::ffi::TensorView prefix_lens,
      const tvm::ffi::TensorView commit_lens,
      const tvm::ffi::TensorView bonus,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView new_seq_lens,
      const tvm::ffi::TensorView new_seq_lens_buf,
      const tvm::ffi::TensorView output_tokens_buf) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    auto B = SymbolicSize{"bs"};
    auto P = SymbolicSize{"pool"};
    auto Q = SymbolicSize{"pool_tokens"};
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(prefix_lens);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device_).verify(commit_lens);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(bonus);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(indices);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device_).verify(new_seq_lens);
    TensorMatcher({P}).with_dtype<int64_t>().with_device(device_).verify(new_seq_lens_buf);
    TensorMatcher({Q}).with_dtype<int64_t>().with_device(device_).verify(output_tokens_buf);
    RuntimeCheck(
        prefix_lens.stride(0) == 1 && commit_lens.stride(0) == 1 && bonus.stride(0) == 1 && indices.stride(0) == 1 &&
            new_seq_lens.stride(0) == 1 && new_seq_lens_buf.stride(0) == 1 && output_tokens_buf.stride(0) == 1,
        "tensors must be contiguous");
    Params p{};
    p.prefix_lens = static_cast<const int64_t*>(prefix_lens.data_ptr());
    p.commit_lens = static_cast<const int32_t*>(commit_lens.data_ptr());
    p.bonus = static_cast<const int64_t*>(bonus.data_ptr());
    p.indices = static_cast<const int64_t*>(indices.data_ptr());
    p.new_seq_lens = static_cast<int64_t*>(new_seq_lens.data_ptr());
    p.new_seq_lens_buf = static_cast<int64_t*>(new_seq_lens_buf.data_ptr());
    p.output_tokens_buf = static_cast<int64_t*>(output_tokens_buf.data_ptr());
    p.bs = static_cast<int>(B.unwrap());
    if (p.bs == 0) return;
    LaunchKernel(1, kThreads, device_.unwrap()).enable_pdl(kUsePDL)(accept_tail_kernel<kUsePDL>, p);
  }
};

}  // namespace sglang::dflash_accept_tail
