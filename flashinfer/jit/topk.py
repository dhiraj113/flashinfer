"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_topk_module() -> JitSpec:
    return gen_jit_spec(
        "topk",
        [
            jit_env.FLASHINFER_CSRC_DIR / "topk.cu",
            jit_env.FLASHINFER_CSRC_DIR / "cub_topk.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_topk_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_fast_topk_clusters_binding.cu",
        ],
        extra_cuda_cflags=["-lineinfo"],
    )


def gen_sglang_dsv4_topk_module() -> JitSpec:
    """Vendored sglang DeepSeek-V4 ragged top-k (benchmark reference backend).

    The vendored sources are C++20 (concepts); nvcc accepts the duplicated
    -std with a warning and uses the last value.  sglang's headers require
    -DSGL_CUDA_ARCH=<__CUDA_ARCH__ value> and static_assert it matches the
    device target, so this module is single-arch: it targets the CURRENT
    device only (fine for a benchmark reference backend).
    """
    import torch

    src_dir = jit_env.FLASHINFER_CSRC_DIR / "sglang_dsv4"
    major, minor = torch.cuda.get_device_capability()
    sgl_arch = major * 100 + minor * 10
    return gen_jit_spec(
        "sglang_dsv4_topk",
        [
            src_dir / "sglang_dsv4_topk.cu",
            src_dir / "flashinfer_sglang_dsv4_topk_binding.cu",
        ],
        extra_cflags=["-std=c++20", f"-DSGL_CUDA_ARCH={sgl_arch}"],
        extra_cuda_cflags=["-std=c++20", "-lineinfo", f"-DSGL_CUDA_ARCH={sgl_arch}"],
        extra_include_paths=[src_dir],
    )
