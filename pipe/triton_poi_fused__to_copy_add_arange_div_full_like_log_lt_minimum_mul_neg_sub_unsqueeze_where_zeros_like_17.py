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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_div_full_like_log_lt_minimum_mul_neg_sub_unsqueeze_where_zeros_like_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 262144}, 'kernel_num_gb': 0.000131072, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_div_full_like_log_lt_minimum_mul_neg_sub_unsqueeze_where_zeros_like_17(out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = (-1)*((0) * ((0) <= (x0 + ((-1)*x1))) + (x0 + ((-1)*x1)) * ((x0 + ((-1)*x1)) < (0)))
        tmp1 = tl.full([1], 16, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = tmp0.to(tl.float32)
        tmp4 = 0.0625
        tmp5 = tmp3 * tmp4
        tmp6 = tl_math.log(tmp5)
        tmp7 = 0.48089834696298783
        tmp8 = tmp6 * tmp7
        tmp9 = 16.0
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10.to(tl.int64)
        tmp12 = tmp11 + tmp1
        tmp13 = tl.full([1], 31, tl.int64)
        tmp14 = tl.minimum(tmp12, tmp13, tl.PropagateNan.ALL)
        tmp15 = tl.where(tmp2, tmp0, tmp14)
        tmp16 = tl.full([1], 0, tl.int64)
        tmp17 = tmp15 + tmp16
        tl.store(out_ptr0 + (x2), tmp17, None)


def get_args():
    arg_0 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    return arg_0, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_div_full_like_log_lt_minimum_mul_neg_sub_unsqueeze_where_zeros_like_17.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_arange_div_full_like_log_lt_minimum_mul_neg_sub_unsqueeze_where_zeros_like_17.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.000131072
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
