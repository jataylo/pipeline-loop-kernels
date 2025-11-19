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
    size_hints={'y': 1024, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'out_ptr1': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_flip_full_like_mul_new_ones_permute_select_slice_transpose_tril_unsqueeze_view_where_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'y': 2101248, 'x': 6566400}, 'kernel_num_gb': 0.00329984, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_flip_full_like_mul_new_ones_permute_select_slice_transpose_tril_unsqueeze_view_where_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 256)
    x2 = xindex
    y1 = yindex // 256
    y3 = yindex
    tmp78 = tl.load(in_ptr2 + (x2 + 513*y3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tmp2.to(tl.int1)
    tmp4 = tl.broadcast_to(x2, [YBLOCK, XBLOCK])
    tmp5 = tl.full([1, 1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1, 1], 256, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp9.to(tl.int1)
    tmp11 = tmp10 & tmp3
    tmp12 = ((((-256) + x2 + 513*y0 + 787968*y1) // 512) % 513)
    tmp13 = tl.full([1, 1], 512, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14.to(tl.int1)
    tmp16 = tmp15 & tmp11
    tmp17 = tl.load(in_ptr0 + (256*(((((-256) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-256) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((((-256) + x2 + 513*y0 + 787968*y1) // 512) % 513))), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (256*(((((-256) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-256) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + ((((-256) + x2 + 513*y0 + 787968*y1) % 512))), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp11, tmp21, tmp22)
    tmp24 = tl.full([1, 1], 0, tl.int64)
    tmp25 = tmp24 >= tmp5
    tmp26 = tmp25.to(tl.int1)
    tmp27 = tmp26 & tmp3
    tmp28 = tl.broadcast_to(x2, [YBLOCK, XBLOCK])
    tmp29 = tl.full([1, 1], 256, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30.to(tl.int1)
    tmp32 = tmp31 & tmp27
    tmp33 = ((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513)
    tmp34 = tl.full([1, 1], 512, tl.int64)
    tmp35 = tmp33 < tmp34
    tmp36 = tmp35.to(tl.int1)
    tmp37 = tmp36 & tmp32
    tmp38 = tl.load(in_ptr0 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513))), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((x2 + 513*y0) % 512))), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp40 = tmp38 * tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp32, tmp42, tmp43)
    tmp45 = tl.load(in_ptr2 + (x2 + 513*y3), tmp27 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp46 = tl.where(tmp30, tmp44, tmp45)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp27, tmp46, tmp47)
    tmp49 = tl.load(in_ptr2 + (x2 + 513*y3), tmp3 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp50 = tl.where(tmp25, tmp48, tmp49)
    tmp51 = tl.where(tmp9, tmp23, tmp50)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp3, tmp51, tmp52)
    tmp54 = tl.full([1, 1], 0, tl.int64)
    tmp55 = tmp54 >= tmp1
    tmp56 = tmp55.to(tl.int1)
    tmp57 = tl.broadcast_to(x2, [YBLOCK, XBLOCK])
    tmp58 = tl.full([1, 1], 256, tl.int64)
    tmp59 = tmp57 < tmp58
    tmp60 = tmp59.to(tl.int1)
    tmp61 = tmp60 & tmp56
    tmp62 = ((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513)
    tmp63 = tl.full([1, 1], 512, tl.int64)
    tmp64 = tmp62 < tmp63
    tmp65 = tmp64.to(tl.int1)
    tmp66 = tmp65 & tmp61
    tmp67 = tl.load(in_ptr0 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513))), tmp66 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp68 = tl.load(in_ptr1 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((x2 + 513*y0) % 512))), tmp66 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp69 = tmp67 * tmp68
    tmp70 = tl.full(tmp69.shape, 0.0, tmp69.dtype)
    tmp71 = tl.where(tmp66, tmp69, tmp70)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp61, tmp71, tmp72)
    tmp74 = tl.load(in_ptr2 + (x2 + 513*y3), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp75 = tl.where(tmp59, tmp73, tmp74)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp56, tmp75, tmp76)
    tmp79 = tl.where(tmp55, tmp77, tmp78)
    tmp80 = tl.where(tmp2, tmp53, tmp79)
    tmp81 = x2
    tmp82 = tl.full([1, 1], 257, tl.int64)
    tmp83 = tmp81 < tmp82
    tmp84 = tmp83.to(tl.int1)
    tmp85 = tl.load(in_ptr3 + (x2 + 257*y0), tmp84 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp86 = (tmp85 != 0)
    tmp87 = tl.full([1, 1], 0, tl.int32)
    tmp88 = tmp87 == tmp87
    tmp89 = tl.full([1, 1], 0, tl.int64)
    tmp90 = tl.full([1, 1], 1, tl.int64)
    tmp91 = tmp89 >= tmp90
    tmp92 = tmp91.to(tl.int1)
    tmp93 = tmp92 & tmp84
    tmp94 = tl.broadcast_to(x2, [YBLOCK, XBLOCK])
    tmp95 = tl.full([1, 1], 256, tl.int64)
    tmp96 = tmp94 < tmp95
    tmp97 = tmp96.to(tl.int1)
    tmp98 = tmp97 & tmp93
    tmp99 = ((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513)
    tmp100 = tl.full([1, 1], 512, tl.int64)
    tmp101 = tmp99 < tmp100
    tmp102 = tmp101.to(tl.int1)
    tmp103 = tmp102 & tmp98
    tmp104 = tl.load(in_ptr0 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((((-131584) + x2 + 513*y0 + 787968*y1) // 512) % 513))), tmp103 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp105 = tl.load(in_ptr1 + (256*(((((-131584) + x2 + 513*y0 + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*y0 + 787968*y1) // 787968) % 4)) + (((x2 + 513*y0) % 512))), tmp103 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp106 = tmp104 * tmp105
    tmp107 = tl.full(tmp106.shape, 0.0, tmp106.dtype)
    tmp108 = tl.where(tmp103, tmp106, tmp107)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp98, tmp108, tmp109)
    tmp111 = tl.load(in_ptr2 + (x2 + 513*y3), tmp93 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp112 = tl.where(tmp96, tmp110, tmp111)
    tmp113 = tl.full(tmp112.shape, 0.0, tmp112.dtype)
    tmp114 = tl.where(tmp93, tmp112, tmp113)
    tmp115 = tl.load(in_ptr2 + (x2 + 513*y3), tmp84 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp116 = tl.where(tmp91, tmp114, tmp115)
    tmp117 = tl.where(tmp88, tmp80, tmp116)
    tmp118 = float("-inf")
    tmp119 = tl.where(tmp86, tmp118, tmp117)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp84, tmp119, tmp120)
    tmp122 = tl.full([1, 1], 0, tl.int32)
    tmp123 = tmp122 == tmp122
    tmp124 = tl.where(tmp123, tmp80, tmp79)
    tmp125 = tl.where(tmp83, tmp121, tmp124)
    tl.store(out_ptr0 + (x2 + 513*y3), tmp80, xmask)
    tl.store(out_ptr1 + (x2 + 513*y3), tmp125, xmask)


def get_args():
    arg_0 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((4, 4, 256, 513), (131328, 525312, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((256, 257), (257, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((4, 256, 513), (131328, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((4, 256, 1, 513), (131328, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1024, 513,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_flip_full_like_mul_new_ones_permute_select_slice_transpose_tril_unsqueeze_view_where_13.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_flip_full_like_mul_new_ones_permute_select_slice_transpose_tril_unsqueeze_view_where_13.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.00329984
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
