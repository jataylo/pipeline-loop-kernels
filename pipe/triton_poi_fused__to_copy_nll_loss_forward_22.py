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
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*i1', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_nll_loss_forward_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'kernel_num_gb': 1.5e-08, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_nll_loss_forward_22(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
        tmp2 = tl.full([1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp4 = tl.full([1], 0, tl.int64)
        tmp5 = tl.where(tmp3, tmp1, tmp4)
        tmp6 = tl.full([R0_BLOCK], 2, tl.int32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp5 < 0
        tmp9 = tl.where(tmp8, tmp7, tmp5)
        tl.device_assert((0 <= tmp9) & (tmp9 < 2), "index out of bounds: 0 <= tmp9 < 2")
        tmp11 = tl.load(in_ptr1 + (tmp9), None, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = -tmp12
        tmp14 = 0.0
        tmp15 = tl.where(tmp3, tmp13, tmp14)
        tmp16 = tmp3.to(tl.int64)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = (tmp15 / tmp17)
        tl.store(out_ptr0 + (tl.full([R0_BLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp3, None)
        tl.store(out_ptr1 + (tl.full([R0_BLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp18, None)


def get_args():
    arg_0 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.bool)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_nll_loss_forward_22.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_nll_loss_forward_22.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 1.5e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
