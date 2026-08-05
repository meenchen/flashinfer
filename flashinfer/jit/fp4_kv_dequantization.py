"""
Copyright (c) 2025 by FlashInfer team.

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
from .core import JitSpec, current_compilation_context, gen_jit_spec


def _gen_fp4_kv_dequantization_module(name: str) -> JitSpec:
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[8, 9, 10, 11, 12]
    )
    return gen_jit_spec(
        name,
        [jit_env.FLASHINFER_CSRC_DIR / "fp4_kv_dequantization.cu"],
        extra_cuda_cflags=nvcc_flags
        + ["-DFLASHINFER_ENABLE_BF16", "-DFLASHINFER_ENABLE_F16"],
    )


def gen_fp4_kv_dequantization_module() -> JitSpec:
    return _gen_fp4_kv_dequantization_module("fp4_kv_dequant")


def gen_nvfp4_kv_pages_to_fp8_module() -> JitSpec:
    return _gen_fp4_kv_dequantization_module("nvfp4_kv_pages_to_fp8")
