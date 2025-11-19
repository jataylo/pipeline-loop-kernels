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
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_embedding_index_select_mul_ne_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'kernel_num_gb': 0.001574912, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_embedding_index_select_mul_ne_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 1024)
        x1 = xindex // 1024
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
        tmp1 = tl.full([R0_BLOCK], 128112, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert((0 <= tmp4) & (tmp4 < 128112), "index out of bounds: 0 <= tmp4 < 128112")
        tmp6 = tl.load(in_ptr1 + (x0 + 1024*tmp4), None)
        tmp7 = 32.0
        tmp8 = tmp6 * tmp7
        tmp10 = tmp9.to(tl.int32)
        tmp11 = tl.full([1], 0, tl.int32)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.full([1], 1, tl.int64)
        tmp14 = tmp0 != tmp13
        tmp15 = tmp14.to(tl.int32)
        tmp16 = tmp12 * tmp15
        tmp17 = tmp16.to(tl.int64)
        tmp18 = tmp17 + tmp13
        tmp19 = tl.full([R0_BLOCK], 1026, tl.int32)
        tmp20 = tmp18 + tmp19
        tmp21 = tmp18 < 0
        tmp22 = tl.where(tmp21, tmp20, tmp18)
        tl.device_assert((0 <= tmp22) & (tmp22 < 1026), "index out of bounds: 0 <= tmp22 < 1026")
        tmp24 = tl.load(in_ptr3 + (x0 + 1024*tmp22), None)
        tmp25 = tmp8 + tmp24
        tl.store(out_ptr0 + (x2), tmp25, None)


def get_args():
    arg_0 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_3 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 131072,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_embedding_index_select_mul_ne_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_embedding_index_select_mul_ne_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.001574912
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
