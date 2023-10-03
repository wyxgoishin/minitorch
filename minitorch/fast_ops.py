from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, jit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index, to_index_by_strides, OutIndex,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


@njit()
def to_index_by_strides(ordinal: int, strides: Strides, out_index: OutIndex) -> OutIndex:
    for idx in range(len(strides)):
        out_index[idx] = ordinal / strides[idx]
        ordinal %= strides[idx]
    return out_index


@njit()
def fast_ndarray_shape_prod(x):
    ret = 1
    for idx in prange(len(x)):
        ret *= x[idx]
    return ret


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"

        # This line JIT compiles your tensor_map
        f = tensor_map(njit()(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        "See `tensor_ops.py`"

        f = tensor_zip(njit()(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
            fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        "See `tensor_ops.py`"
        f = tensor_reduce(njit()(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """
        Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
            a : tensor data a
            b : tensor data b

        Returns:
            New tensor data
        """

        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
        fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
            out: Storage,
            out_shape: Shape,
            out_strides: Strides,
            in_storage: Storage,
            in_shape: Shape,
            in_strides: Strides,
    ) -> None:
        out_size = fast_ndarray_shape_prod(out_shape)
        for out_idx in prange(out_size):
            out_index = np.zeros_like(out_shape)
            to_index_by_strides(out_idx, out_strides, out_index)
            in_index = np.zeros_like(in_shape)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_idx = index_to_position(in_index, in_strides)
            out[out_idx] = fn(in_storage[in_idx])

    return njit(parallel=True)(_map)  # type: ignore


def tensor_zip(
        fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
            out: Storage,
            out_shape: Shape,
            out_strides: Strides,
            a_storage: Storage,
            a_shape: Shape,
            a_strides: Strides,
            b_storage: Storage,
            b_shape: Shape,
            b_strides: Strides,
    ) -> None:
        out_size = fast_ndarray_shape_prod(out_shape)
        for out_idx in prange(out_size):
            out_index = np.zeros_like(out_shape)
            to_index_by_strides(out_idx, out_strides, out_index)

            in_index_a = np.zeros_like(a_shape)
            broadcast_index(out_index, out_shape, a_shape, in_index_a)
            in_idx_a = index_to_position(in_index_a, a_strides)

            in_index_b = np.zeros_like(b_shape)
            broadcast_index(out_index, out_shape, b_shape, in_index_b)
            in_idx_b = index_to_position(in_index_b, b_strides)

            out[out_idx] = fn(a_storage[in_idx_a], b_storage[in_idx_b])

    return njit(parallel=True)(_zip)  # type: ignore


def tensor_reduce(
        fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
            out: Storage,
            out_shape: Shape,
            out_strides: Strides,
            a_storage: Storage,
            a_shape: Shape,
            a_strides: Strides,
            reduce_dim: int,
    ) -> None:
        out_size = fast_ndarray_shape_prod(out_shape)
        for i in prange(out_size):
            out_index = np.zeros_like(out_shape)
            to_index_by_strides(i, out_strides, out_index)

            for j in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = j
                a_idx = index_to_position(out_index, a_strides)
                out[i] = fn(out[i], a_storage[a_idx])

    return njit(parallel=True)(_reduce)  # type: ignore


def _tensor_matrix_multiply(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
) -> None:
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    """
    This task actually requires implementing torch.matmul of Tensor A and B
        Tensor A has shape [a1, a2, ..., an]
        Tensor B has shape [b1, b2, ..., bn]
    Condition:
        1. an = bn - 1, namely A.shape[-1] = B.shape[-2] 
        (because in matrix multiplication, last is view as col, least last is view as row)
        2. [a1, a2, ..., an-2] and [b1, b2, ..., bn-2] are broadcastable
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    out_size = fast_ndarray_shape_prod(out_shape)
    for i in prange(out_size):
        out_index = np.zeros_like(out_shape)
        to_index_by_strides(i, out_strides, out_index)
        a_index = np.zeros_like(a_shape)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        b_index = np.zeros_like(b_shape)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        for j in range(a_shape[-1]):
            a_index[-1] = j
            a_idx = index_to_position(a_index, a_strides)

            b_index[-2] = j
            b_idx = index_to_position(b_index, b_strides)

            out[i] += a_storage[a_idx] * b_storage[b_idx]


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
