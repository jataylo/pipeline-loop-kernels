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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_full_like_mul_permute_slice_transpose_unsqueeze_view_where_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'y': 8404992, 'x': 22063104}, 'kernel_num_gb': 0.014856704, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_full_like_mul_permute_slice_transpose_unsqueeze_view_where_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    tl.static_assert(YBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, YBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        yindex = yoffset + r + lanes[:, None]
        ymask  = yindex < ynumel
        x2 = xindex
        y0 = (yindex % 1024)
        y1 = yindex // 1024
        y3 = yindex
        tmp8 = tl.load(in_ptr1 + (x2 + 513*((y0 % 256)) + 131328*y1), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp33 = tl.load(in_ptr4 + (x2 + 513*((y0 % 256)) + 131328*y1 + 525312*(y0 // 256)), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp0 = y0
        tmp1 = tl.full([1, 1], 256, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = tl.load(in_ptr0 + (x2 + 513*y0 + 131328*y1), tmp3 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = y0 // 256
        tmp6 = tl.full([1, 1], 0, tl.int32)
        tmp7 = tmp5 == tmp6
        tmp9 = tl.full([1, 1], 1, tl.int64)
        tmp10 = tmp5 >= tmp9
        tmp11 = tmp10.to(tl.int1)
        tmp12 = tl.broadcast_to(x2, [R0_BLOCK, XBLOCK])
        tmp13 = tl.full([1, 1], 256, tl.int64)
        tmp14 = tmp12 < tmp13
        tmp15 = tmp14.to(tl.int1)
        tmp16 = tmp15 & tmp11
        tmp17 = ((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 512) % 513)
        tmp18 = tl.full([1, 1], 512, tl.int64)
        tmp19 = tmp17 < tmp18
        tmp20 = tmp19.to(tl.int1)
        tmp21 = tmp20 & tmp16
        tmp22 = tl.load(in_ptr2 + (256*(((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 787968) % 4)) + (((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 512) % 513))), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (256*(((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 262656) % 3)) + 1024*(((((-131584) + x2 + 513*((y0 % 256)) + 262656*(y0 // 256) + 787968*y1) // 787968) % 4)) + (((x2 + 513*((y0 % 256))) % 512))), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
        tmp26 = tl.where(tmp21, tmp24, tmp25)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp16, tmp26, tmp27)
        tmp29 = tl.load(in_ptr4 + (x2 + 513*((y0 % 256)) + 131328*y1 + 525312*(y0 // 256)), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.where(tmp14, tmp28, tmp29)
        tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
        tmp32 = tl.where(tmp11, tmp30, tmp31)
        tmp34 = tl.where(tmp10, tmp32, tmp33)
        tmp35 = tl.where(tmp7, tmp8, tmp34)
        tmp36 = tl.where(tmp2, tmp4, tmp35)
        tmp37 = tl.full([1, 1], 768, tl.int64)
        tmp38 = tmp0 >= tmp37
        tmp39 = tmp38.to(tl.int1)
        tmp40 = tl.broadcast_to(x2, [R0_BLOCK, XBLOCK])
        tmp41 = tl.full([1, 1], 256, tl.int64)
        tmp42 = tmp40 >= tmp41
        tmp43 = tmp42.to(tl.int1)
        tmp44 = tmp43 & tmp39
        tmp45 = tl.load(in_ptr5 + ((-197632) + x2 + 257*y0), tmp44 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp46 = (tmp45 != 0)
        tmp47 = float("-inf")
        tmp48 = tl.where(tmp46, tmp47, tmp36)
        tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
        tmp50 = tl.where(tmp44, tmp48, tmp49)
        tmp51 = tl.where(tmp42, tmp50, tmp36)
        tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
        tmp53 = tl.where(tmp39, tmp51, tmp52)
        tmp54 = tl.where(tmp38, tmp53, tmp36)
        tl.debug_barrier()
        tl.store(in_out_ptr0 + (x2 + 513*y3), tmp54, xmask)


def get_args():
    arg_0 = rand_strided((4, 1024, 1, 513), (525312, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((4, 256, 1, 513), (131328, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((4, 256, 513), (131328, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((4, 2, 512, 1), (1024, 512, 1, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((4, 4, 256, 513), (131328, 525312, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_6 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 4096, 513,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_full_like_mul_permute_slice_transpose_unsqueeze_view_where_14.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_as_strided_constant_pad_nd_copy_expand_full_like_mul_permute_slice_transpose_unsqueeze_view_where_14.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.014856704
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
