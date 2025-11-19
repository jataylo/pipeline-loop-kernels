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
    size_hints={'y': 4096, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_as_strided_constant_pad_nd_copy_mul_new_zeros_permute_select_slice_unsqueeze_view_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'y': 2101248, 'x': 13658112}, 'kernel_num_gb': 0.00421888, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_as_strided_constant_pad_nd_copy_mul_new_zeros_permute_select_slice_unsqueeze_view_12(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 513
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y2 = yindex // 1024
    x3 = xindex
    y0 = (yindex % 256)
    y1 = ((yindex // 256) % 4)
    y5 = yindex
    tmp0 = y2
    tmp1 = tl.full([1, 1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x3
    tmp4 = tl.full([1, 1], 256, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp5.to(tl.int1)
    tmp7 = (((656384 + x3 + 513*y0) // 512) % 513)
    tmp8 = tl.full([1, 1], 512, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9.to(tl.int1)
    tmp11 = tmp10 & tmp6
    tmp12 = tl.load(in_ptr0 + (256*((656384 + x3 + 513*y0) // 262656) + 1024*y1 + 1024*((656384 + x3 + 513*y0) // 787968) + ((((656384 + x3 + 513*y0) // 512) % 513))), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr1 + (256*((656384 + x3 + 513*y0) // 262656) + 1024*y1 + 1024*((656384 + x3 + 513*y0) // 787968) + (((x3 + 513*y0) % 512))), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp6, tmp16, tmp17)
    tmp19 = tl.full([1, 1], 3, tl.int64)
    tmp20 = tmp19 < tmp19
    tmp21 = tmp20.to(tl.int1)
    tmp22 = tl.broadcast_to(x3, [YBLOCK, XBLOCK])
    tmp23 = tl.full([1, 1], 256, tl.int64)
    tmp24 = tmp22 >= tmp23
    tmp25 = tmp24.to(tl.int1)
    tmp26 = tmp25 & tmp21
    tmp27 = (((787712 + x3 + 513*y0) // 512) % 513)
    tmp28 = tl.full([1, 1], 512, tl.int64)
    tmp29 = tmp27 < tmp28
    tmp30 = tmp29.to(tl.int1)
    tmp31 = tmp30 & tmp26
    tmp32 = tl.load(in_ptr0 + (256*((((787712 + x3 + 513*y0) // 262656) % 3)) + 1024*((((787712 + x3 + 513*y0 + 787968*y1) // 787968) % 4)) + ((((787712 + x3 + 513*y0) // 512) % 513))), tmp31 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tl.load(in_ptr1 + (256*((((787712 + x3 + 513*y0) // 262656) % 3)) + 1024*((((787712 + x3 + 513*y0 + 787968*y1) // 787968) % 4)) + (((787712 + x3 + 513*y0) % 512))), tmp31 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp31, tmp34, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp26, tmp36, tmp37)
    tmp39 = 0.0
    tmp40 = tl.where(tmp24, tmp38, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp21, tmp40, tmp41)
    tmp43 = 0.0
    tmp44 = tl.where(tmp20, tmp42, tmp43)
    tmp45 = tl.where(tmp5, tmp18, tmp44)
    tmp46 = tmp0 < tmp19
    tmp47 = tmp46.to(tl.int1)
    tmp48 = tl.broadcast_to(x3, [YBLOCK, XBLOCK])
    tmp49 = tl.full([1, 1], 256, tl.int64)
    tmp50 = tmp48 >= tmp49
    tmp51 = tmp50.to(tl.int1)
    tmp52 = tmp51 & tmp47
    tmp53 = ((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 512) % 513)
    tmp54 = tl.full([1, 1], 512, tl.int64)
    tmp55 = tmp53 < tmp54
    tmp56 = tmp55.to(tl.int1)
    tmp57 = tmp56 & tmp52
    tmp58 = tl.load(in_ptr0 + (256*(((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 262656) % 3)) + 1024*(((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 787968) % 4)) + (((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 512) % 513))), tmp57 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tl.load(in_ptr1 + (256*(((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 262656) % 3)) + 1024*(((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) // 787968) % 4)) + ((((-256) + x3 + 513*y0 + 262656*y2 + 787968*y1) % 512))), tmp57 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp60 = tmp58 * tmp59
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp57, tmp60, tmp61)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp52, tmp62, tmp63)
    tmp65 = 0.0
    tmp66 = tl.where(tmp50, tmp64, tmp65)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp47, tmp66, tmp67)
    tmp69 = tl.where(tmp46, tmp68, tmp43)
    tmp70 = tl.where(tmp2, tmp45, tmp69)
    tl.store(out_ptr0 + (x3 + 513*y5), tmp70, xmask)


def get_args():
    arg_0 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((4, 4, 256, 513), (131328, 525312, 513, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 4096, 513,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_as_strided_constant_pad_nd_copy_mul_new_zeros_permute_select_slice_unsqueeze_view_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_as_strided_constant_pad_nd_copy_mul_new_zeros_permute_select_slice_unsqueeze_view_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.00421888
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
