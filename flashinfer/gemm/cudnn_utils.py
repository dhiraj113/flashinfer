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

"""Shared cuDNN utilities for GEMM operations.

This module contains cuDNN handle/availability helpers, tensor-shape utilities,
and data-type converters that are used by both :mod:`gemm_base` and
:mod:`gemm_bias` (and any future ``gemm_<op>`` modules).  Keeping them here
avoids circular imports between those sibling modules.
"""

from enum import Enum

import torch

# ---------------------------------------------------------------------------
# cuDNN optional import
# ---------------------------------------------------------------------------

CUDNN_AVAILABLE = False
try:
    import cudnn

    CUDNN_AVAILABLE = True
except ImportError:
    pass
except OSError as e:
    error_msg = str(e).lower()
    is_lib_missing = any(ext in error_msg for ext in [".so", ".dll"])
    if not is_lib_missing:
        raise


# ---------------------------------------------------------------------------
# UID enum for cuDNN graph tensor identifiers
# ---------------------------------------------------------------------------


class UIDs(Enum):
    """UIDs for CUDNN graph tensors"""

    A_UID = 0
    B_UID = 1
    ALPHA_UID = 2
    BLOCK_DESCALE_A_UID = 3
    BLOCK_DESCALE_B_UID = 4
    A_SCALE_UID = 5
    B_SCALE_UID = 6
    BIAS_UID = 7
    O_UID = 8


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------


def _check_cudnn_availability():
    """Check if cuDNN is available and raise exception if not."""
    if not CUDNN_AVAILABLE:
        raise RuntimeError(
            "cuDNN is not available. Please install cuDNN to use FP8 GEMM functions. "
            "You can install it with: pip install nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )


def _check_cudnn_fp4_availability():
    """Check if cuDNN FP4 support is available and raise exception if not."""
    _check_cudnn_availability()

    # Check cuDNN version for FP4 support (requires 1.13.* or later)
    try:
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])

        if (major, minor) < (1, 13):
            raise RuntimeError(
                f"cuDNN FP4 requires version 1.13+, found {version_str}. "
                f"Upgrade: pip install --upgrade nvidia-cudnn-cu12 nvidia-cudnn-frontend"
            )
    except (ImportError, AttributeError, ValueError, IndexError) as e:
        raise RuntimeError(
            "Unable to determine cuDNN version. FP4 requires cuDNN 1.13+."
        ) from e

    # Check cuDNN backend version for FP4 support (requires >= 91002)
    try:
        backend_version = cudnn.backend_version()
        if backend_version < 91002:
            raise RuntimeError(
                f"cuDNN FP4 requires backend version >= 91002, found {backend_version}. "
                f"Please upgrade cuDNN backend."
            )
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            "Unable to determine cuDNN backend version. FP4 requires backend >= 91002."
        ) from e


def _cudnn_available_or_raise_for_backend(backend):
    # When cudnn is not available:
    # Return False for auto backend or raise error for explicit cuDNN backend.
    if CUDNN_AVAILABLE:
        return True
    if backend == "cudnn":
        _check_cudnn_availability()
    return False


def _is_cublas_fp4_available_in_cudnn():
    """Check if cuBLAS backend for FP4 GEMM is available in cuDNN."""

    # Check cuDNN backend version for FP4 support (requires cudnn_version == 9.11.1 or cudnn_version >= 9.13)
    backend_version = cudnn.backend_version()
    CUDNN_VERSION_9_11_1 = 91101
    CUDNN_VERSION_9_13_0 = 91300
    return (
        backend_version == CUDNN_VERSION_9_11_1
        or backend_version >= CUDNN_VERSION_9_13_0
    )


def _check_cudnn_override_shape_availability():
    """Raise if the installed cuDNN backend does not support is_override_shape_enabled."""
    _check_cudnn_availability()
    backend_version = cudnn.backend_version()
    if backend_version < 92100:
        raise RuntimeError(
            f"cuDNN override-shape GEMM requires backend version >= 92100 (9.21.0), "
            f"found {backend_version}. "
            f"Please upgrade cuDNN: pip install --upgrade nvidia-cudnn-cu12 nvidia-cudnn-frontend"
        )
    try:
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])
        required_frontend_version = (1, 24) if backend_version >= 92300 else (1, 20)
        if (major, minor) < required_frontend_version:
            raise RuntimeError(
                f"cuDNN override-shape GEMM requires cudnn-frontend version >= "
                f"{required_frontend_version[0]}.{required_frontend_version[1]}, found {version_str}. "
                f"Please upgrade: pip install --upgrade nvidia-cudnn-frontend"
            )
    except (AttributeError, ValueError, IndexError) as e:
        raise RuntimeError(
            "Unable to determine cudnn-frontend version. "
            "Override-shape GEMM requires cudnn-frontend >= 1.20, or >= 1.24 with cuDNN backend >= 9.23.0"
        ) from e


def is_cudnn_override_shape_available() -> bool:
    """Return True if the installed cuDNN backend supports is_override_shape_enabled."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        backend_version = cudnn.backend_version()
        if backend_version < 92100:
            return False
        version_str = cudnn.__version__
        major, minor = map(int, version_str.split(".")[:2])
        required_frontend_version = (1, 24) if backend_version >= 92300 else (1, 20)
        return (major, minor) >= required_frontend_version
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Workspace size helpers
# ---------------------------------------------------------------------------


def _get_cudnn_workspace_size(graph, tactic: int) -> int:
    if tactic < 0:
        return graph.get_workspace_size()
    return graph.get_workspace_size_plan_at_index(tactic)


def _get_cudnn_override_shape_workspace_size(
    graph,
    tactic: int,
    cudnn_handle,
    override_uids,
    override_shapes,
    override_strides,
) -> int:
    if cudnn.backend_version() >= 92300:
        if tactic < 0:
            return graph.get_workspace_size(
                cudnn_handle, override_uids, override_shapes, override_strides
            )
        return graph.get_workspace_size_plan_at_index(
            tactic, cudnn_handle, override_uids, override_shapes, override_strides
        )
    else:
        if tactic < 0:
            return graph.get_workspace_size()
        return graph.get_workspace_size_plan_at_index(tactic)


# ---------------------------------------------------------------------------
# cuDNN handle (one per GPU device)
# ---------------------------------------------------------------------------

_cudnn_handles: dict[int, int] = {}


def _get_cudnn_handle(device, stream: torch.cuda.Stream):
    """Create and return a cached cuDNN handle."""
    global _cudnn_handles
    device_id = device.index

    if _cudnn_handles.get(device_id) is None:
        _check_cudnn_availability()
        _cudnn_handles[device_id] = cudnn.create_handle()
        print("cudnn_handle created for device_id = {}\n".format(device_id))
    cudnn.set_stream(_cudnn_handles[device_id], stream.cuda_stream)

    return _cudnn_handles[device_id]


# ---------------------------------------------------------------------------
# Data-type conversion
# ---------------------------------------------------------------------------


def _torch_data_type_to_cudnn_data_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif dtype == torch.float16:
        return cudnn.data_type.HALF
    elif dtype == torch.float32:
        return cudnn.data_type.FLOAT
    elif dtype == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    elif dtype == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# Tensor shape / stride helpers
# ---------------------------------------------------------------------------


def _get_bf16_3d_shape_stride(tensor: torch.Tensor):
    """Expand 2d tensor to 3d tensor for cuDNN"""
    if tensor.dim() != 2 and tensor.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D tensor")
    shape = list(tensor.shape)
    stride = list(tensor.stride())

    if len(shape) == 2:
        shape.insert(0, 1)
        stride.insert(0, tensor.numel())

    return (tuple(shape), tuple(stride))


def _get_real_fp4_shape_from_packed_uint8(packed_fp4_tensor):
    # the FP4 data are packed into uint8, we need to expand the shape and stride information to get the real shape and stride to be used in the cuDNN graph.
    is_column_major = packed_fp4_tensor.stride(-2) == 1
    real_shape = list(packed_fp4_tensor.shape)
    real_stride = list(packed_fp4_tensor.stride())

    # this function will be used for both mm and bmm, so we need to insert batch dimension if the tensor is 2d
    if len(real_shape) == 2:
        real_shape.insert(0, 1)
        real_stride.insert(0, packed_fp4_tensor.numel())

    # each packed uint8 contains 2 fp4 elements
    real_shape[-2 if is_column_major else -1] *= 2
    if is_column_major:
        real_stride[-1] *= 2
        for i in range(len(real_stride) - 2):
            real_stride[i] *= 2
    else:
        for i in range(len(real_stride) - 1):
            real_stride[i] *= 2

    return (tuple(real_shape), tuple(real_stride))


def _expand_block_scale_tensor_shape(block_scale_tensor, batch_size):
    # This function will be shared for both mm and bmm, when 2d block scale tensor is provided, we need unfold the batch dimension. the unfoled dim and stride is returned.
    block_scale_shape = list(block_scale_tensor.shape)
    block_scale_stride = list(block_scale_tensor.stride())

    if len(block_scale_shape) == 2:
        # expand to 3d
        block_scale_shape.insert(0, batch_size)
        block_scale_stride.insert(0, 1)

        # update the stride and shape for the expanded dimension
        is_column_major = block_scale_tensor.stride(-2) == 1
        expand_dim = 2 if is_column_major else 1

        assert block_scale_shape[expand_dim] % batch_size == 0
        block_scale_shape[expand_dim] = block_scale_shape[expand_dim] // batch_size
        block_scale_stride[0] = (
            block_scale_stride[expand_dim] * block_scale_shape[expand_dim]
        )
    elif len(block_scale_shape) == 3:
        pass
    else:
        raise ValueError(
            f"Unsupported block scale tensor shape: {block_scale_shape}, expected 2d or 3d."
        )

    return (tuple(block_scale_shape), tuple(block_scale_stride))
