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
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {'load_seed_offset': 1}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_addmm_native_dropout_view_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 59771904}, 'kernel_num_gb': 0.04404328, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_addmm_native_dropout_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = xindex
        x1 = (xindex % 768)
        tmp6 = tl.load(in_ptr1 + (x0), None)
        tmp7 = tl.load(in_ptr2 + (x0), None)
        tmp9 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
        tmp16 = tl.load(in_ptr4 + (x0), None).to(tl.float32)
        tmp17 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = x0
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.1
        tmp5 = tmp3 > tmp4
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = 1.1111111111111112
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp6 + tmp13
        tmp15 = tmp5.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp16 + tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tmp20 * tmp11
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp14 + tmp22
        tl.store(out_ptr1 + (x0), tmp5, None)
        tl.store(out_ptr2 + (x0), tmp23, None)


def get_args():
    arg_0 = rand_strided((2,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((2, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 2048, 768), (1572864, 768, 1), device='cuda:0', dtype=torch.bool)
    arg_3 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.bool)
    arg_7 = rand_strided((4096, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_8 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 3145728,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_addmm_native_dropout_view_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_addmm_native_dropout_view_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.04404328
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
