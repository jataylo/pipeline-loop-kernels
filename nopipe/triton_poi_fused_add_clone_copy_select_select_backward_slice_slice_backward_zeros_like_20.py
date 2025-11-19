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
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_copy_select_select_backward_slice_slice_backward_zeros_like_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 94556160}, 'kernel_num_gb': 0.050429952, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_copy_select_select_backward_slice_slice_backward_zeros_like_20(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9455616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 6156) % 512)
    x0 = (xindex % 513)
    x3 = xindex // 3151872
    x5 = (xindex % 6156)
    x7 = xindex // 6156
    x1 = ((xindex // 513) % 12)
    tmp67 = tl.load(in_ptr1 + (x0 + 513*x2 + 262656*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 255, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 511, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.int1)
    tmp7 = x0
    tmp8 = tl.full([1], 257, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp9.to(tl.int1)
    tmp11 = tmp10 & tmp6
    tmp12 = 1 + x3
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tmp12 == tmp13
    tmp15 = (-255) + x2
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tmp17.to(tl.int1)
    tmp19 = tmp18 & tmp11
    tmp20 = (-257) + x0
    tmp21 = tl.full([1], 1, tl.int64)
    tmp22 = tmp20 >= tmp21
    tmp23 = tl.full([1], 256, tl.int64)
    tmp24 = tmp20 < tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tmp25.to(tl.int1)
    tmp27 = tmp26 & tmp19
    tmp28 = 0.0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.load(in_ptr0 + ((-1583297) + x5 + 6208*x2), tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp25, tmp30, tmp31)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp19, tmp32, tmp33)
    tmp35 = tl.load(in_ptr0 + ((-1583297) + x5 + 6208*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp17, tmp34, tmp35)
    tmp37 = tl.load(in_ptr0 + (5951 + x5 + 6208*x2 + 1589248*x3), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.where(tmp14, tmp36, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp11, tmp38, tmp39)
    tmp41 = 0.0
    tmp42 = tl.where(tmp9, tmp40, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp6, tmp42, tmp43)
    tmp45 = 0.0
    tmp46 = tl.where(tmp5, tmp44, tmp45)
    tmp47 = x3
    tmp48 = tl.full([1], 0, tl.int32)
    tmp49 = tmp47 == tmp48
    tmp50 = tmp0 < tmp1
    tmp51 = tmp50.to(tl.int1)
    tmp52 = x0
    tmp53 = tl.full([1], 258, tl.int64)
    tmp54 = tmp52 >= tmp53
    tmp55 = tmp54.to(tl.int1)
    tmp56 = tmp55 & tmp51
    tmp57 = tl.load(in_ptr0 + (5951 + x5 + 6208*x2), tmp56 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp58 = 0.0
    tmp59 = tl.where(tmp54, tmp57, tmp58)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp51, tmp59, tmp60)
    tmp62 = tl.where(tmp50, tmp61, tmp45)
    tmp63 = tl.where(tmp49, tmp62, tmp45)
    tmp64 = tmp63 + tmp46
    tmp65 = tl.full([1], 2, tl.int32)
    tmp66 = tmp47 == tmp65
    tmp68 = tl.where(tmp66, tmp67, tmp45)
    tmp69 = tmp64 + tmp68
    tmp70 = tl.full([1], 256, tl.int64)
    tmp71 = tmp0 < tmp70
    tmp72 = tmp71.to(tl.int1)
    tmp73 = x0
    tmp74 = tl.full([1], 257, tl.int64)
    tmp75 = tmp73 < tmp74
    tmp76 = tmp75.to(tl.int1)
    tmp77 = tmp76 & tmp72
    tmp78 = tl.load(in_ptr2 + (256 + x5 + 6208*x2 + 1589248*x3), tmp77 & xmask, other=0.0).to(tl.float32)
    tmp79 = 0.0
    tmp80 = tl.where(tmp75, tmp78, tmp79)
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp72, tmp80, tmp81)
    tmp83 = tl.where(tmp71, tmp82, tmp45)
    tmp84 = tmp69 + tmp83
    tl.store(out_ptr1 + (x0 + 513*x7 + 787968*x1), tmp84, xmask)


def get_args():
    arg_0 = rand_strided((12, 4, 256, 513), (513, 1589248, 6208, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((12, 512, 513), (262656, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((12, 4, 256, 513), (513, 1589248, 6208, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((12, 3, 512, 513), (787968, 262656, 513, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, 9455616,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_copy_select_select_backward_slice_slice_backward_zeros_like_20.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_add_clone_copy_select_select_backward_slice_slice_backward_zeros_like_20.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.050429952
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
