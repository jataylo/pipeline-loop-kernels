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
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy__unsafe_view_add_clone_constant_pad_nd_exp_masked_fill_native_dropout_sub_transpose_unsqueeze_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 340623360}, 'kernel_num_gb': 0.155938816, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax__to_copy__unsafe_view_add_clone_constant_pad_nd_exp_masked_fill_native_dropout_sub_transpose_unsqueeze_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37847040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 770)
    x1 = ((xindex // 770) % 48)
    x2 = xindex // 36960
    tmp0 = x0
    tmp1 = tl.full([1], 513, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tmp2.to(tl.int1)
    tmp4 = tl.load(in_ptr0 + (x0 + 513*((x1 % 12)) + 6272*x2 + 6422528*(x1 // 12)), tmp3, other=0.0)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2 + 1024*(x1 // 12)), tmp3, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (x0 + 513*x2 + 525312*x1), tmp3, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0 + 513*x2 + 525312*(x1 // 12)), tmp3, other=0.0).to(tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (12*x2 + 12288*(x1 // 12) + ((x1 % 12))), tmp3, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 - tmp11
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tl.load(in_ptr5 + (12*x2 + 12288*(x1 // 12) + ((x1 % 12))), tmp3, eviction_policy='evict_last', other=0.0)
    tmp15 = (tmp13 / tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp6, tmp16, tmp15)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp5 * tmp18
    tmp20 = 1.1111111111111112
    tmp21 = tmp19 * tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp3, tmp21, tmp22)
    tl.store(out_ptr0 + (x0 + 770*x2 + 788480*x1), tmp23, None)


def get_args():
    arg_0 = rand_strided((4, 1024, 12, 513), (6422528, 6272, 513, 1), device='cuda:0', dtype=torch.bool)
    arg_1 = rand_strided((4, 1024), (1024, 1), device='cuda:0', dtype=torch.bool)
    arg_2 = rand_strided((4, 1024, 12, 513), (6303744, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((4, 1024, 1, 513), (525312, 513, 525312, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((4, 1024, 12, 1), (12288, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((4, 1024, 12, 1), (12288, 12, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((48, 4, 256, 770), (788480, 197120, 770, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 37847040,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax__to_copy__unsafe_view_add_clone_constant_pad_nd_exp_masked_fill_native_dropout_sub_transpose_unsqueeze_18.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__softmax__to_copy__unsafe_view_add_clone_constant_pad_nd_exp_masked_fill_native_dropout_sub_transpose_unsqueeze_18.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.155938816
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
