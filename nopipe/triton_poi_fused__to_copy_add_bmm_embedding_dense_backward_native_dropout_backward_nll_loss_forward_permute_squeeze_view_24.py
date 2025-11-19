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
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_bmm_embedding_dense_backward_native_dropout_backward_nll_loss_forward_permute_squeeze_view_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 9961472}, 'kernel_num_gb': 0.007868416, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_bmm_embedding_dense_backward_native_dropout_backward_nll_loss_forward_permute_squeeze_view_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr0 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (x2), None)
    tmp1 = tl.full([1], -1, tl.int64)
    tmp2 = tmp0 == tmp1
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = 1.1111111111111112
    tmp16 = tmp14 * tmp15
    tmp17 = tmp12 * tmp16
    tmp18 = 0.0
    tmp19 = tl.where(tmp2, tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp19, None)


def get_args():
    arg_0 = rand_strided((512, 1, 1024), (1024, 524288, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 1), (1, 512), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_3 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_4 = rand_strided((512, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg_5 = rand_strided((512, 1, 1024), (1024, 1024, 1), device='cuda:0', dtype=torch.bool)
    return arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 524288,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_bmm_embedding_dense_backward_native_dropout_backward_nll_loss_forward_permute_squeeze_view_24.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_bmm_embedding_dense_backward_native_dropout_backward_nll_loss_forward_permute_squeeze_view_24.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.007868416
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
