# KERNEL CALLS: 1

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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_arange_bitwise_and_eq_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'kernel_num_gb': 0.00845824, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_arange_bitwise_and_eq_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 512)
    x0 = (xindex % 512)
    x2 = xindex // 262144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp1 <= tmp0
    tmp3 = tl.full([1], True, tl.int1)
    tmp4 = tmp3 & tmp2
    tmp5 = tl.full([XBLOCK], 512, tl.int32)
    tmp6 = tmp0 + tmp5
    tmp7 = tmp0 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp0)
    tl.device_assert((0 <= tmp8) & (tmp8 < 512), "index out of bounds: 0 <= tmp8 < 512")
    tmp10 = tl.load(in_ptr1 + (tmp8 + 512*x2), None, eviction_policy='evict_last')
    tmp12 = tmp10 == tmp11
    tmp13 = tmp4 & tmp12
    tmp14 = 0.0
    tmp15 = float("-inf")
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)


def get_args():
    arg_0 = rand_strided((512,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((16, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((16, 1, 512, 512), (262144, 262144, 512, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_arange_bitwise_and_eq_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_arange_bitwise_and_eq_expand_index_le_new_ones_scalar_tensor_unsqueeze_where_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.00845824
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
