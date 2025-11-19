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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_clone_im2col_transpose_unsqueeze_view_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'kernel_num_gb': 0.003970568, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_clone_im2col_transpose_unsqueeze_view_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 9)
    x2 = xindex // 3456
    x1 = ((xindex // 9) % 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x2 + 512*x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp1 = tl.full([XBLOCK], 520, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 520), "index out of bounds: 0 <= tmp4 < 520")
    tmp8 = tl.full([XBLOCK], 1, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 1), "index out of bounds: 0 <= tmp11 < 1")
    tmp13 = (-4) + tmp4
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tl.full([1], 512, tl.int64)
    tmp18 = tmp14 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tmp19.to(tl.int1)
    tmp21 = tl.load(in_ptr2 + ((-1536) + x1 + 384*tmp4), tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp20, tmp24, tmp25)
    tl.store(out_ptr0 + (x3), tmp26, None)


def get_args():
    arg_0 = rand_strided((9, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((512, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 512, 384, 9), (1769472, 3456, 9, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 1769472,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_clone_im2col_transpose_unsqueeze_view_12.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_addmm_clone_im2col_transpose_unsqueeze_view_12.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.003970568
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
