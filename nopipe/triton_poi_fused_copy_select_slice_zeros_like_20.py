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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_select_slice_zeros_like_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 176504832}, 'kernel_num_gb': 0.100859904, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_select_slice_zeros_like_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 131328) % 4)
    x0 = (xindex % 513)
    x1 = ((xindex // 513) % 256)
    x3 = xindex // 525312
    x4 = (xindex % 131328)
    x5 = xindex
    tmp72 = tl.load(in_ptr0 + (x4 + 525312*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp74 = tl.load(in_ptr0 + (393984 + x4 + 525312*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp119 = tl.load(in_ptr0 + (x5), None).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5.to(tl.int1)
    tmp7 = 0.0
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp12.to(tl.int1)
    tmp14 = x0
    tmp15 = tl.full([1], 256, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16.to(tl.int1)
    tmp18 = tmp17 & tmp13
    tmp19 = 0.0
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.full([1], 3, tl.int32)
    tmp23 = tl.full([1], 0, tl.int32)
    tmp24 = tmp22 == tmp23
    tmp25 = x1
    tmp26 = tl.full([1], 1, tl.int64)
    tmp27 = tmp25 >= tmp26
    tmp28 = tmp27.to(tl.int1)
    tmp29 = tmp28 & tmp13
    tmp30 = x0
    tmp31 = tl.full([1], 1, tl.int64)
    tmp32 = tmp30 >= tmp31
    tmp33 = tl.full([1], 256, tl.int64)
    tmp34 = tmp30 < tmp33
    tmp35 = tmp32 & tmp34
    tmp36 = tmp35.to(tl.int1)
    tmp37 = tmp36 & tmp29
    tmp38 = 0.0
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.load(in_ptr0 + (x4 + 525312*x3), tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.where(tmp35, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp29, tmp42, tmp43)
    tmp45 = tl.load(in_ptr0 + (x4 + 525312*x3), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tl.where(tmp27, tmp44, tmp45)
    tmp47 = tl.load(in_ptr0 + (393984 + x4 + 525312*x3), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp48 = tl.where(tmp24, tmp46, tmp47)
    tmp49 = tl.where(tmp16, tmp21, tmp48)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp13, tmp49, tmp50)
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = tmp1 == tmp52
    tmp54 = x1
    tmp55 = tmp54 >= tmp11
    tmp56 = tmp55.to(tl.int1)
    tmp57 = x0
    tmp58 = tl.full([1], 1, tl.int64)
    tmp59 = tmp57 >= tmp58
    tmp60 = tl.full([1], 256, tl.int64)
    tmp61 = tmp57 < tmp60
    tmp62 = tmp59 & tmp61
    tmp63 = tmp62.to(tl.int1)
    tmp64 = tmp63 & tmp56
    tmp65 = 0.0
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp64, tmp65, tmp66)
    tmp68 = tl.load(in_ptr0 + (x4 + 525312*x3), tmp56, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp69 = tl.where(tmp62, tmp67, tmp68)
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp56, tmp69, tmp70)
    tmp73 = tl.where(tmp55, tmp71, tmp72)
    tmp75 = tl.where(tmp53, tmp73, tmp74)
    tmp76 = tl.where(tmp12, tmp51, tmp75)
    tmp77 = tl.where(tmp5, tmp9, tmp76)
    tmp78 = tmp0 >= tmp11
    tmp79 = tmp78.to(tl.int1)
    tmp80 = x0
    tmp81 = tl.full([1], 256, tl.int64)
    tmp82 = tmp80 < tmp81
    tmp83 = tmp82.to(tl.int1)
    tmp84 = tmp83 & tmp79
    tmp85 = 0.0
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp84, tmp85, tmp86)
    tmp88 = x2
    tmp89 = tl.full([1], 0, tl.int32)
    tmp90 = tmp88 == tmp89
    tmp91 = x1
    tmp92 = tl.full([1], 1, tl.int64)
    tmp93 = tmp91 >= tmp92
    tmp94 = tmp93.to(tl.int1)
    tmp95 = tmp94 & tmp79
    tmp96 = x0
    tmp97 = tl.full([1], 1, tl.int64)
    tmp98 = tmp96 >= tmp97
    tmp99 = tl.full([1], 256, tl.int64)
    tmp100 = tmp96 < tmp99
    tmp101 = tmp98 & tmp100
    tmp102 = tmp101.to(tl.int1)
    tmp103 = tmp102 & tmp95
    tmp104 = 0.0
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp103, tmp104, tmp105)
    tmp107 = tl.load(in_ptr0 + (x4 + 525312*x3), tmp95, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp108 = tl.where(tmp101, tmp106, tmp107)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp95, tmp108, tmp109)
    tmp111 = tl.load(in_ptr0 + (x4 + 525312*x3), tmp79, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp112 = tl.where(tmp93, tmp110, tmp111)
    tmp113 = tl.load(in_ptr0 + (x5), tmp79, other=0.0).to(tl.float32)
    tmp114 = tl.where(tmp90, tmp112, tmp113)
    tmp115 = tl.where(tmp82, tmp87, tmp114)
    tmp116 = tl.full(tmp115.shape, 0.0, tmp115.dtype)
    tmp117 = tl.where(tmp79, tmp115, tmp116)
    tmp118 = tmp0 == tmp52
    tmp120 = tl.where(tmp118, tmp73, tmp119)
    tmp121 = tl.where(tmp78, tmp117, tmp120)
    tmp122 = tl.where(tmp2, tmp77, tmp121)
    tl.store(out_ptr0 + (x5), tmp122, None)


def get_args():
    arg_0 = rand_strided((48, 4, 256, 513), (525312, 131328, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((48, 4, 256, 513), (525312, 131328, 513, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, 25214976,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_select_slice_zeros_like_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_copy_select_slice_zeros_like_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.100859904
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
