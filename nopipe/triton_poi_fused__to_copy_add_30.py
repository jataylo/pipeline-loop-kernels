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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp16', 'in_ptr6': '*fp16', 'in_ptr7': '*fp16', 'in_ptr8': '*fp16', 'in_ptr9': '*fp16', 'in_ptr10': '*fp16', 'in_ptr11': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 2147483648}, 'kernel_num_gb': 2.147483648, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67108864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (x0), None).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (x0), None).to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (x0), None).to(tl.float32)
    tmp23 = tl.load(in_ptr8 + (x0), None).to(tl.float32)
    tmp26 = tl.load(in_ptr9 + (x0), None).to(tl.float32)
    tmp29 = tl.load(in_ptr10 + (x0), None).to(tl.float32)
    tmp32 = tl.load(in_ptr11 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 + tmp30
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 + tmp33
    tl.store(in_out_ptr0 + (x0), tmp34, None)


def get_args():
    arg_0 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_6 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_7 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_8 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_9 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_10 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_11 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    arg_12 = rand_strided((4096, 16384), (16384, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, 67108864,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_30.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_30.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 2.147483648
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
