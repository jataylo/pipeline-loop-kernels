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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_copy_permute_select_slice_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 239542272}, 'kernel_num_gb': 0.25214976, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_copy_permute_select_slice_view_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        x5 = xindex // 24624
        tmp69 = tl.load(in_ptr1 + (x4 + 98560*x2), None, eviction_policy='evict_last').to(tl.float32)
        tmp91 = tl.load(in_ptr1 + (x4 + 24640*x3 + 98560*x2), None).to(tl.float32)
        tmp0 = x3
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = x2
        tmp4 = tl.full([1], 1, tl.int64)
        tmp5 = tmp3 >= tmp4
        tmp6 = tmp5.to(tl.int1)
        tmp7 = x0
        tmp8 = tl.full([1], 1, tl.int64)
        tmp9 = tmp7 >= tmp8
        tmp10 = tl.full([1], 256, tl.int64)
        tmp11 = tmp7 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = tmp12.to(tl.int1)
        tmp14 = tmp13 & tmp6
        tmp15 = ((((-256) + x0 + 513*x2 + 787968*x1) // 512) % 513)
        tmp16 = tl.full([1], 512, tl.int64)
        tmp17 = tmp15 < tmp16
        tmp18 = tmp17.to(tl.int1)
        tmp19 = tmp18 & tmp14
        tmp20 = tl.load(in_ptr0 + (262144*(((((-256) + x0 + 513*x2 + 787968*x1) // 262656) % 144)) + ((((-256) + x0 + 513*x2 + 787968*x1) % 262656))), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
        tmp22 = tl.where(tmp14, tmp20, tmp21)
        tmp23 = tl.full([1], 0, tl.int64)
        tmp24 = tmp23 >= tmp8
        tmp25 = tmp24.to(tl.int1)
        tmp26 = tmp25 & tmp6
        tmp27 = x0
        tmp28 = tl.full([1], 256, tl.int64)
        tmp29 = tmp27 < tmp28
        tmp30 = tmp29.to(tl.int1)
        tmp31 = tmp30 & tmp26
        tmp32 = ((((-131584) + x0 + 513*x2 + 787968*x1) // 512) % 513)
        tmp33 = tl.full([1], 512, tl.int64)
        tmp34 = tmp32 < tmp33
        tmp35 = tmp34.to(tl.int1)
        tmp36 = tmp35 & tmp31
        tmp37 = tl.load(in_ptr0 + (512*(((((-131584) + x0 + 513*x2 + 787968*x1) // 512) % 513)) + 262144*(((((-131584) + x0 + 513*x2 + 787968*x1) // 262656) % 144)) + (((x0 + 513*x2) % 512))), tmp36, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp31, tmp37, tmp38)
        tmp40 = tl.load(in_ptr1 + (x4 + 98560*x2), tmp26, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp41 = tl.where(tmp29, tmp39, tmp40)
        tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
        tmp43 = tl.where(tmp26, tmp41, tmp42)
        tmp44 = tl.load(in_ptr1 + (x4 + 98560*x2), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tl.where(tmp24, tmp43, tmp44)
        tmp46 = tl.where(tmp12, tmp22, tmp45)
        tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
        tmp48 = tl.where(tmp6, tmp46, tmp47)
        tmp49 = tl.full([1], 0, tl.int64)
        tmp50 = tmp49 >= tmp4
        tmp51 = tmp50.to(tl.int1)
        tmp52 = x0
        tmp53 = tl.full([1], 256, tl.int64)
        tmp54 = tmp52 < tmp53
        tmp55 = tmp54.to(tl.int1)
        tmp56 = tmp55 & tmp51
        tmp57 = ((((-131584) + x0 + 513*x2 + 787968*x1) // 512) % 513)
        tmp58 = tl.full([1], 512, tl.int64)
        tmp59 = tmp57 < tmp58
        tmp60 = tmp59.to(tl.int1)
        tmp61 = tmp60 & tmp56
        tmp62 = tl.load(in_ptr0 + (512*(((((-131584) + x0 + 513*x2 + 787968*x1) // 512) % 513)) + 262144*(((((-131584) + x0 + 513*x2 + 787968*x1) // 262656) % 144)) + (((x0 + 513*x2) % 512))), tmp61, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
        tmp64 = tl.where(tmp56, tmp62, tmp63)
        tmp65 = tl.load(in_ptr1 + (x4 + 98560*x2), tmp51, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp66 = tl.where(tmp54, tmp64, tmp65)
        tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
        tmp68 = tl.where(tmp51, tmp66, tmp67)
        tmp70 = tl.where(tmp50, tmp68, tmp69)
        tmp71 = tl.where(tmp5, tmp48, tmp70)
        tmp72 = tmp0 >= tmp4
        tmp73 = tmp72.to(tl.int1)
        tmp74 = x0
        tmp75 = tl.full([1], 256, tl.int64)
        tmp76 = tmp74 < tmp75
        tmp77 = tmp76.to(tl.int1)
        tmp78 = tmp77 & tmp73
        tmp79 = ((((-131584) + x0 + 513*x2 + 262656*x3 + 787968*x1) // 512) % 513)
        tmp80 = tl.full([1], 512, tl.int64)
        tmp81 = tmp79 < tmp80
        tmp82 = tmp81.to(tl.int1)
        tmp83 = tmp82 & tmp78
        tmp84 = tl.load(in_ptr0 + (512*(((((-131584) + x0 + 513*x2 + 262656*x3 + 787968*x1) // 512) % 513)) + 262144*(((((-131584) + x0 + 513*x2 + 262656*x3 + 787968*x1) // 262656) % 144)) + (((x0 + 513*x2) % 512))), tmp83, other=0.0).to(tl.float32)
        tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
        tmp86 = tl.where(tmp78, tmp84, tmp85)
        tmp87 = tl.load(in_ptr1 + (x4 + 24640*x3 + 98560*x2), tmp73, other=0.0).to(tl.float32)
        tmp88 = tl.where(tmp76, tmp86, tmp87)
        tmp89 = tl.full(tmp88.shape, 0.0, tmp88.dtype)
        tmp90 = tl.where(tmp73, tmp88, tmp89)
        tmp92 = tl.where(tmp72, tmp90, tmp91)
        tmp93 = tl.where(tmp2, tmp71, tmp92)
        tl.store(out_ptr0 + (x0 + 513*x5 + 525312*x1), tmp93, None)


def get_args():
    arg_0 = rand_strided((144, 512, 512), (262144, 512, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((48, 4, 256, 513), (513, 24640, 98560, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((48, 4, 256, 513), (525312, 131328, 513, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 25214976,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_copy_permute_select_slice_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_constant_pad_nd_copy_permute_select_slice_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.25214976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
