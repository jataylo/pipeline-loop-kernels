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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'xnumel': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_nll_loss_forward_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'kernel_num_gb': 5.2e-08, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_nll_loss_forward_19(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
        tmp5 = tl.load(in_ptr0 + (1))
        tmp6 = tl.broadcast_to(tmp5, [R0_BLOCK])
        tmp10 = tl.load(in_ptr0 + (2))
        tmp11 = tl.broadcast_to(tmp10, [R0_BLOCK])
        tmp15 = tl.load(in_ptr0 + (3))
        tmp16 = tl.broadcast_to(tmp15, [R0_BLOCK])
        tmp2 = tl.full([1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp4 = tmp3.to(tl.int64)
        tmp7 = tmp6 != tmp2
        tmp8 = tmp7.to(tl.int64)
        tmp9 = tmp4 + tmp8
        tmp12 = tmp11 != tmp2
        tmp13 = tmp12.to(tl.int64)
        tmp14 = tmp9 + tmp13
        tmp17 = tmp16 != tmp2
        tmp18 = tmp17.to(tl.int64)
        tmp19 = tmp14 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1], 0, tl.int64)
        tmp22 = tl.where(tmp3, tmp1, tmp21)
        tmp23 = tl.full([R0_BLOCK], 2, tl.int32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp22 < 0
        tmp26 = tl.where(tmp25, tmp24, tmp22)
        tl.device_assert((0 <= tmp26) & (tmp26 < 2), "index out of bounds: 0 <= tmp26 < 2")
        tmp28 = tl.load(in_ptr1 + (tmp26), None, eviction_policy='evict_last').to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = -tmp29
        tmp31 = 0.0
        tmp32 = tl.where(tmp3, tmp30, tmp31)
        tmp33 = tl.where(tmp7, tmp6, tmp21)
        tmp34 = tmp33 + tmp23
        tmp35 = tmp33 < 0
        tmp36 = tl.where(tmp35, tmp34, tmp33)
        tl.device_assert((0 <= tmp36) & (tmp36 < 2), "index out of bounds: 0 <= tmp36 < 2")
        tmp38 = tl.load(in_ptr1 + (2 + tmp36), None, eviction_policy='evict_last').to(tl.float32)
        tmp39 = tmp38.to(tl.float32)
        tmp40 = -tmp39
        tmp41 = tl.where(tmp7, tmp40, tmp31)
        tmp42 = tmp32 + tmp41
        tmp43 = tl.where(tmp12, tmp11, tmp21)
        tmp44 = tmp43 + tmp23
        tmp45 = tmp43 < 0
        tmp46 = tl.where(tmp45, tmp44, tmp43)
        tl.device_assert((0 <= tmp46) & (tmp46 < 2), "index out of bounds: 0 <= tmp46 < 2")
        tmp48 = tl.load(in_ptr1 + (4 + tmp46), None, eviction_policy='evict_last').to(tl.float32)
        tmp49 = tmp48.to(tl.float32)
        tmp50 = -tmp49
        tmp51 = tl.where(tmp12, tmp50, tmp31)
        tmp52 = tmp42 + tmp51
        tmp53 = tl.where(tmp17, tmp16, tmp21)
        tmp54 = tmp53 + tmp23
        tmp55 = tmp53 < 0
        tmp56 = tl.where(tmp55, tmp54, tmp53)
        tl.device_assert((0 <= tmp56) & (tmp56 < 2), "index out of bounds: 0 <= tmp56 < 2")
        tmp58 = tl.load(in_ptr1 + (6 + tmp56), None, eviction_policy='evict_last').to(tl.float32)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = -tmp59
        tmp61 = tl.where(tmp17, tmp60, tmp31)
        tmp62 = tmp52 + tmp61
        tmp63 = (tmp62 / tmp20)
        tl.store(out_ptr0 + (tl.full([R0_BLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp20, None)
        tl.store(in_out_ptr0 + (tl.full([R0_BLOCK], 0, tl.int32).broadcast_to(XBLOCK)), tmp63, None)


def get_args():
    arg_0 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((4, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 1,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_nll_loss_forward_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_nll_loss_forward_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 5.2e-08
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
