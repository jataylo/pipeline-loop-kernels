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
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_slice_backward_view_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 229376}, 'kernel_num_gb': 0.000164864, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_slice_backward_view_26(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 128)
        x1 = xindex // 128
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
        tmp3 = tl.load(in_ptr1 + (128 + x0 + 384*x1), None).to(tl.float32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 == tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = x1
        tmp6 = tl.full([1], 127, tl.int64)
        tmp7 = tmp5 < tmp6
        tmp8 = tmp7.to(tl.int1)
        tmp9 = 1 + x1
        tmp10 = tl.full([1], 0, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tmp11.to(tl.int1)
        tmp13 = tmp12 & tmp8
        tmp14 = tl.load(in_ptr1 + (640 + x0 + 384*x1), tmp13, other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
        tmp17 = tl.where(tmp13, tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp8, tmp17, tmp18)
        tmp20 = 0.0
        tmp21 = tl.where(tmp7, tmp19, tmp20)
        tmp22 = tmp4 + tmp21
        tmp23 = tl.full([1], 1, tl.int64)
        tmp24 = tmp5 >= tmp23
        tmp25 = tmp24.to(tl.int1)
        tmp26 = (-1) + x1
        tmp27 = tl.full([1], 128, tl.int64)
        tmp28 = tmp26 < tmp27
        tmp29 = tmp28.to(tl.int1)
        tmp30 = tmp29 & tmp25
        tmp31 = tl.load(in_ptr1 + ((-384) + x0 + 384*x1), tmp30, other=0.0).to(tl.float32)
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
        tmp34 = tl.where(tmp30, tmp32, tmp33)
        tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
        tmp36 = tl.where(tmp25, tmp34, tmp35)
        tmp37 = tl.where(tmp24, tmp36, tmp20)
        tmp38 = tmp22 + tmp37
        tmp39 = tl.where(tmp2, tmp20, tmp38)
        tl.store(out_ptr0 + (x2), tmp39, None)


def get_args():
    arg_0 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((128, 384), (384, 1), device='cuda:0', dtype=torch.float16)
    arg_2 = rand_strided((1, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, 16384,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_slice_backward_view_26.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_add_constant_pad_nd_embedding_dense_backward_nll_loss_forward_slice_slice_backward_view_26.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.000164864
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
