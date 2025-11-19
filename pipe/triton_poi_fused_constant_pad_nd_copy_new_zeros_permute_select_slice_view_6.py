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
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_copy_new_zeros_permute_select_slice_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 176504832}, 'kernel_num_gb': 0.201719808, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_copy_new_zeros_permute_select_slice_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 513)
        x1 = ((xindex // 513) % 48)
        x2 = ((xindex // 24624) % 256)
        x3 = xindex // 6303744
        x4 = (xindex % 24624)
        tmp0 = x3
        tmp1 = tl.full([1], 3, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = x0
        tmp4 = tl.full([1], 256, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5.to(tl.int1)
        tmp7 = (((656384 + x0 + 513*x2) // 512) % 513)
        tmp8 = tl.full([1], 512, tl.int64)
        tmp9 = tmp7 < tmp8
        tmp10 = tmp9.to(tl.int1)
        tmp11 = tmp10 & tmp6
        tmp12 = tl.load(in_ptr0 + (512*((((656384 + x0 + 513*x2) // 512) % 513)) + 262144*((656384 + x0 + 513*x2) // 262656) + 786432*x1 + 786432*((656384 + x0 + 513*x2) // 787968) + (((x0 + 513*x2) % 512))), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
        tmp14 = tl.where(tmp6, tmp12, tmp13)
        tmp15 = tl.full([1], 3, tl.int64)
        tmp16 = tmp15 < tmp15
        tmp17 = tmp16.to(tl.int1)
        tmp18 = x0
        tmp19 = tl.full([1], 256, tl.int64)
        tmp20 = tmp18 >= tmp19
        tmp21 = tmp20.to(tl.int1)
        tmp22 = tmp21 & tmp17
        tmp23 = (((787712 + x0 + 513*x2) // 512) % 513)
        tmp24 = tl.full([1], 512, tl.int64)
        tmp25 = tmp23 < tmp24
        tmp26 = tmp25.to(tl.int1)
        tmp27 = tmp26 & tmp22
        tmp28 = tl.load(in_ptr0 + (262144*((((787712 + x0 + 513*x2) // 262656) % 3)) + 786432*((((787712 + x0 + 513*x2 + 787968*x1) // 787968) % 48)) + (((787712 + x0 + 513*x2) % 262656))), tmp27, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
        tmp30 = tl.where(tmp22, tmp28, tmp29)
        tmp31 = 0.0
        tmp32 = tl.where(tmp20, tmp30, tmp31)
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp17, tmp32, tmp33)
        tmp35 = 0.0
        tmp36 = tl.where(tmp16, tmp34, tmp35)
        tmp37 = tl.where(tmp5, tmp14, tmp36)
        tmp38 = tmp0 < tmp15
        tmp39 = tmp38.to(tl.int1)
        tmp40 = x0
        tmp41 = tl.full([1], 256, tl.int64)
        tmp42 = tmp40 >= tmp41
        tmp43 = tmp42.to(tl.int1)
        tmp44 = tmp43 & tmp39
        tmp45 = ((((-256) + x0 + 513*x2 + 262656*x3 + 787968*x1) // 512) % 513)
        tmp46 = tl.full([1], 512, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tmp47.to(tl.int1)
        tmp49 = tmp48 & tmp44
        tmp50 = tl.load(in_ptr0 + (262144*(((((-256) + x0 + 513*x2 + 262656*x3 + 787968*x1) // 262656) % 144)) + ((((-256) + x0 + 513*x2 + 262656*x3 + 787968*x1) % 262656))), tmp49, other=0.0).to(tl.float32)
        tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
        tmp52 = tl.where(tmp44, tmp50, tmp51)
        tmp53 = 0.0
        tmp54 = tl.where(tmp42, tmp52, tmp53)
        tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
        tmp56 = tl.where(tmp39, tmp54, tmp55)
        tmp57 = tl.where(tmp38, tmp56, tmp35)
        tmp58 = tl.where(tmp2, tmp37, tmp57)
        tl.store(out_ptr0 + (x4 + 24640*x3 + 98560*x2), tmp58, None)


def get_args():
    arg_0 = rand_strided((144, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((48, 4, 256, 513), (513, 24640, 98560, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, 25214976,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_copy_new_zeros_permute_select_slice_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_constant_pad_nd_copy_new_zeros_permute_select_slice_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.201719808
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
