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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr1': '*i1', 'out_ptr2': '*fp16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_embedding_native_dropout_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'kernel_num_gb': 0.029393688, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_embedding_native_dropout_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x2 = xindex // 1024
    x1 = (xindex % 1024)
    tmp6 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tl.full([XBLOCK], 32000, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert((0 <= tmp10) & (tmp10 < 32000), "index out of bounds: 0 <= tmp10 < 32000")
    tmp12 = tl.load(in_ptr2 + (x1 + 1024*tmp10), None)
    tmp13 = tmp5 * tmp12
    tmp14 = 1.1111111111111112
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, None)
    tl.store(out_ptr2 + (x0), tmp16, None)


def get_args():
    arg_0 = rand_strided((99,), (1,), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((512, 8), (8, 1), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((512, 8, 1024), (8192, 1024, 1), device='cuda:0', dtype=torch.bool)
    arg_4 = rand_strided((512, 8, 1024), (8192, 1024, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = 0
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 4194304,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_embedding_native_dropout_1.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_embedding_native_dropout_1.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.029393688
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
