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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_embedding_index_select_mul_native_dropout_ne_view_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'kernel_num_gb': 0.02310964, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_embedding_index_select_mul_native_dropout_ne_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = xindex // 1024
    x1 = (xindex % 1024)
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.full([XBLOCK], 128112, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 128112), "index out of bounds: 0 <= tmp10 < 128112")
    tmp12 = tl.load(in_ptr2 + (x1 + 1024*tmp10), None)
    tmp13 = 32.0
    tmp14 = tmp12 * tmp13
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1], 1, tl.int64)
    tmp20 = tmp6 != tmp19
    tmp21 = tmp20.to(tl.int32)
    tmp22 = tmp18 * tmp21
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23 + tmp19
    tmp25 = tl.full([XBLOCK], 1026, tl.int32)
    tmp26 = tmp24 + tmp25
    tmp27 = tmp24 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp24)
    tl.device_assert((0 <= tmp28) & (tmp28 < 1026), "index out of bounds: 0 <= tmp28 < 1026")
    tmp30 = tl.load(in_ptr4 + (x1 + 1024*tmp28), None)
    tmp31 = tmp14 + tmp30
    tmp32 = tmp5 * tmp31
    tmp33 = 1.1111111111111112
    tmp34 = tmp32 * tmp33
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp34, None)


def get_args():
    arg_0 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((128112, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_4 = rand_strided((1026, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg_6 = rand_strided((16, 128, 1024), (131072, 1024, 1), device='cuda:0', dtype=torch.float32)
    arg_7 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 2097152,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_embedding_index_select_mul_native_dropout_ne_view_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_embedding_index_select_mul_native_dropout_ne_view_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.02310964
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
