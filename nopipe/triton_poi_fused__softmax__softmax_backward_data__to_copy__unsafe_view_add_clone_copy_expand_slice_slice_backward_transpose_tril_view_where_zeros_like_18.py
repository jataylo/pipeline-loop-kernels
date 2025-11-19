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
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 153391104}, 'kernel_num_gb': 0.101123072, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25214976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 6156) % 1024)
    x0 = (xindex % 513)
    x3 = xindex // 6303744
    x4 = (xindex % 6156)
    x1 = ((xindex // 513) % 12)
    tmp0 = x2
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp2.to(tl.int1)
    tmp4 = x0
    tmp5 = tl.full([1], 257, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tmp6.to(tl.int1)
    tmp8 = tmp7 & tmp3
    tmp9 = tl.load(in_ptr0 + (x0 + 257*x2), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = (tmp9 != 0)
    tmp11 = x2
    tmp12 = tl.full([1], 768, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tmp13.to(tl.int1)
    tmp15 = tmp14 & tmp8
    tmp16 = x0
    tmp17 = tl.full([1], 256, tl.int64)
    tmp18 = tmp16 >= tmp17
    tmp19 = tmp18.to(tl.int1)
    tmp20 = tmp19 & tmp15
    tmp21 = 0.0
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.load(in_ptr1 + (x4 + 6208*x2 + 3178496*(((x2 % 256)) // 256) + 6356992*x3), tmp15, other=0.0).to(tl.float32)
    tmp25 = tl.where(tmp18, tmp23, tmp24)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp15, tmp25, tmp26)
    tmp28 = tl.load(in_ptr1 + (x4 + 6208*x2 + 3178496*(((x2 % 256)) // 256) + 6356992*x3), tmp8, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp13, tmp27, tmp28)
    tmp30 = tl.load(in_ptr2 + ((-197632) + x0 + 257*x2), tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = (tmp30 != 0)
    tmp32 = tl.load(in_ptr1 + (x4 + 6208*x2 + 3178496*(((x2 % 256)) // 256) + 6356992*x3), tmp20, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp31, tmp21, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp20, tmp33, tmp34)
    tmp36 = 0.0
    tmp37 = tl.where(tmp18, tmp35, tmp36)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp15, tmp37, tmp38)
    tmp40 = 0.0
    tmp41 = tl.where(tmp13, tmp39, tmp40)
    tmp42 = tmp29 + tmp41
    tmp43 = tl.where(tmp10, tmp40, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp8, tmp43, tmp44)
    tmp46 = 0.0
    tmp47 = tl.where(tmp6, tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp3, tmp47, tmp48)
    tmp50 = 0.0
    tmp51 = tl.where(tmp2, tmp49, tmp50)
    tl.store(out_ptr0 + (x0 + 513*x2 + 525312*x1 + 6303744*x3), tmp51, None)


def get_args():
    arg_0 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((4, 12, 1024, 513), (6356992, 513, 6208, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((1, 256, 1, 257), (65792, 257, 257, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((4, 1024, 12, 513), (6303744, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, 25214976,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax__softmax_backward_data__to_copy__unsafe_view_add_clone_copy_expand_slice_slice_backward_transpose_tril_view_where_zeros_like_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.101123072
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
