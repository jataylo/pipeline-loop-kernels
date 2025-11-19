# KERNEL CALLS: 1
# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/hf_baseline/25/c25b3igmlh4ukapf5lf3kaf4ttem36skpkdgcdvtu6xyah2rfy3y.py
# Topologically Sorted Source Nodes: [mask_cond, add, view, lt, masked_fill_, mask], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full]
# Source node to ATen node mapping:
#   add => add
#   lt => lt
#   mask => full_default
#   mask_cond => iota
#   masked_fill_ => full_default_1, where
#   view => view
# Graph fragment:
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 1), kwargs = {})
#   %view : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [128, 1]), kwargs = {})
#   %lt : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota, %view), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 128], -3.4028234663852886e+38), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "f32[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %full_default_1, %full_default), kwargs = {})
#   return %where
triton_poi_fused_add_arange_full_lt_masked_fill_view_0 = async_compile.triton('triton_poi_fused_add_arange_full_lt_masked_fill_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_full_lt_masked_fill_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 131072}, 'kernel_num_gb': 6.5536e-05, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_full_lt_masked_fill_view_0(out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 128)
        x1 = xindex // 128
        x2 = xindex
        tmp0 = x0
        tmp1 = 1 + x1
        tmp2 = tmp0 < tmp1
        tmp3 = 0.0
        tmp4 = -3.4028234663852886e+38
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tl.store(out_ptr0 + (x2), tmp5, None)


def get_args():
    arg_0 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_arange_full_lt_masked_fill_view_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_arange_full_lt_masked_fill_view_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 6.5536e-05
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((128, 128), (128, 1), torch.float32)
            # Topologically Sorted Source Nodes: [mask_cond, add, view, lt, masked_fill_, mask], Original ATen: [aten.arange, aten.add, aten.view, aten.lt, aten.masked_fill, aten.full]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_arange_full_lt_masked_fill_view_0.run(buf0, 16384, stream=stream0)
        return (reinterpret_tensor(buf0, (1, 1, 128, 128), (16384, 16384, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    fn = lambda: call([])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XGLMForCausalLM', benchmark_compiled_module)
