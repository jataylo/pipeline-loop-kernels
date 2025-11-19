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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 202770432}, 'kernel_num_gb': 0.15142144, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 513)
        x1 = ((xindex // 513) % 1024)
        x2 = xindex // 525312
        x3 = xindex
        tmp62 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), None).to(tl.float32)
        tmp78 = tl.load(in_out_ptr0 + (x3), None).to(tl.float32)
        tmp0 = x1
        tmp1 = tl.full([1], 256, tl.int64)
        tmp2 = tmp0 < tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = x0
        tmp5 = tl.full([1], 257, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.int1)
        tmp8 = tmp7 & tmp3
        tmp9 = 0.0
        tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
        tmp11 = tl.where(tmp8, tmp9, tmp10)
        tmp12 = x1
        tmp13 = tl.full([1], 768, tl.int64)
        tmp14 = tmp12 >= tmp13
        tmp15 = tmp14.to(tl.int1)
        tmp16 = tmp15 & tmp3
        tmp17 = x0
        tmp18 = tl.full([1], 256, tl.int64)
        tmp19 = tmp17 >= tmp18
        tmp20 = tmp19.to(tl.int1)
        tmp21 = tmp20 & tmp16
        tmp22 = 0.0
        tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
        tmp24 = tl.where(tmp21, tmp22, tmp23)
        tmp25 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), tmp16, other=0.0).to(tl.float32)
        tmp26 = tl.where(tmp19, tmp24, tmp25)
        tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
        tmp28 = tl.where(tmp16, tmp26, tmp27)
        tmp29 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), tmp3, other=0.0).to(tl.float32)
        tmp30 = tl.where(tmp14, tmp28, tmp29)
        tmp31 = tl.load(in_ptr1 + ((-197632) + x0 + 257*x1), tmp21, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = (tmp31 != 0)
        tmp33 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), tmp21, other=0.0).to(tl.float32)
        tmp34 = tl.where(tmp32, tmp22, tmp33)
        tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
        tmp36 = tl.where(tmp21, tmp34, tmp35)
        tmp37 = 0.0
        tmp38 = tl.where(tmp19, tmp36, tmp37)
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp16, tmp38, tmp39)
        tmp41 = 0.0
        tmp42 = tl.where(tmp14, tmp40, tmp41)
        tmp43 = tmp30 + tmp42
        tmp44 = tl.where(tmp6, tmp11, tmp43)
        tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
        tmp46 = tl.where(tmp3, tmp44, tmp45)
        tmp47 = tl.full([1], 768, tl.int64)
        tmp48 = tmp0 >= tmp47
        tmp49 = tmp48.to(tl.int1)
        tmp50 = x0
        tmp51 = tl.full([1], 256, tl.int64)
        tmp52 = tmp50 >= tmp51
        tmp53 = tmp52.to(tl.int1)
        tmp54 = tmp53 & tmp49
        tmp55 = 0.0
        tmp56 = tl.full(tmp55.shape, 0.0, tmp55.dtype)
        tmp57 = tl.where(tmp54, tmp55, tmp56)
        tmp58 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), tmp49, other=0.0).to(tl.float32)
        tmp59 = tl.where(tmp52, tmp57, tmp58)
        tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
        tmp61 = tl.where(tmp49, tmp59, tmp60)
        tmp63 = tl.where(tmp48, tmp61, tmp62)
        tmp64 = tl.load(in_ptr1 + ((-197632) + x0 + 257*x1), tmp54, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp65 = (tmp64 != 0)
        tmp66 = tl.load(in_ptr0 + (x0 + 513*((x2 % 12)) + 6208*x1 + 6356992*(x2 // 12) + 19070976*(((x2 % 12)) // 12)), tmp54, other=0.0).to(tl.float32)
        tmp67 = tl.where(tmp65, tmp55, tmp66)
        tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
        tmp69 = tl.where(tmp54, tmp67, tmp68)
        tmp70 = 0.0
        tmp71 = tl.where(tmp52, tmp69, tmp70)
        tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
        tmp73 = tl.where(tmp49, tmp71, tmp72)
        tmp74 = 0.0
        tmp75 = tl.where(tmp48, tmp73, tmp74)
        tmp76 = tmp63 + tmp75
        tmp77 = tl.where(tmp2, tmp46, tmp76)
        tmp79 = tmp77 + tmp78
        tl.store(in_out_ptr0 + (x3), tmp79, None)


def get_args():
    arg_0 = rand_strided((48, 4, 256, 513), (525312, 131328, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((4, 12, 1024, 513), (6356992, 513, 6208, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 25214976,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_19.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_19.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.15142144
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
