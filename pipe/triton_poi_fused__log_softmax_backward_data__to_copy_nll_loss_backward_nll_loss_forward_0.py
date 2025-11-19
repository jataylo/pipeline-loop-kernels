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
    size_hints={'x': 2}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 3}, 'kernel_num_gb': 2.1e-08, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
        tmp11 = tl.load(in_ptr1 + (0))
        tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
        tmp13 = tl.load(in_ptr2 + (0))
        tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
        tmp20 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
        tmp2 = tl.full([1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp4 = tl.full([1], 0, tl.int64)
        tmp5 = tl.where(tmp3, tmp1, tmp4)
        tmp6 = x0
        tmp7 = tmp5 == tmp6
        tmp8 = -1.0
        tmp9 = 0.0
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp15 = tmp14.to(tl.int64)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = (tmp12 / tmp16)
        tmp18 = tl.where(tmp3, tmp17, tmp9)
        tmp19 = tmp10 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp22 = libdevice.exp(tmp21)
        tmp23 = tmp5 == tmp4
        tmp24 = tl.where(tmp23, tmp8, tmp9)
        tmp25 = tmp24 * tmp18
        tmp26 = tl.full([1], 1, tl.int64)
        tmp27 = tmp5 == tmp26
        tmp28 = tl.where(tmp27, tmp8, tmp9)
        tmp29 = tmp28 * tmp18
        tmp30 = tmp25 + tmp29
        tmp31 = tmp22 * tmp30
        tmp32 = tmp19 - tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(in_out_ptr0 + (x0), tmp33, xmask)


def get_args():
    arg_0 = rand_strided((1, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.bool)
    return arg_0, arg_1, arg_2, arg_3, 2,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_0.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 2.1e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
