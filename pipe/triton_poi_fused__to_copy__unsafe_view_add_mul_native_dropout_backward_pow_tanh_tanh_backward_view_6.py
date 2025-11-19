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
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*i1', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 31457280}, 'kernel_num_gb': 0.027262976, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0), None)
        tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 1.1111111111111112
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp8 = 0.5
        tmp9 = tmp7 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp7.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tmp12 * tmp11
        tmp14 = 0.044715
        tmp15 = tmp13 * tmp14
        tmp16 = tmp11 + tmp15
        tmp17 = 0.7978845608028654
        tmp18 = tmp16 * tmp17
        tmp19 = libdevice.tanh(tmp18)
        tmp20 = 1.0
        tmp21 = tmp19 + tmp20
        tmp22 = tmp10 * tmp21
        tmp23 = tmp6 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp6 * tmp26
        tmp28 = tmp27 * tmp10
        tmp29 = tmp19 * tmp19
        tmp30 = tmp20 - tmp29
        tmp31 = tmp28 * tmp30
        tmp32 = tmp31 * tmp17
        tmp33 = tmp32 * tmp14
        tmp34 = 3.0
        tmp35 = tmp12 * tmp34
        tmp36 = tmp33 * tmp35
        tmp37 = tmp32.to(tl.float32)
        tmp38 = tmp36.to(tl.float32)
        tmp39 = tmp37 + tmp38
        tmp40 = tmp27 * tmp21
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp41 * tmp8
        tmp43 = tmp39 + tmp42
        tl.store(out_ptr0 + (x0), tmp24, None)
        tl.store(in_out_ptr0 + (x0), tmp43, None)


def get_args():
    arg_0 = rand_strided((16, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((16, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg_3 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((2048, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((16, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 2097152,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_6.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy__unsafe_view_add_mul_native_dropout_backward_pow_tanh_tanh_backward_view_6.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.027262976
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
