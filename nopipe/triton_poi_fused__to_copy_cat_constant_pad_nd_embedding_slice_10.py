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
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_constant_pad_nd_embedding_slice_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'kernel_num_gb': 0.028341248, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_constant_pad_nd_embedding_slice_10(in_ptr0, in_ptr1, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 128)
    x4 = xindex // 384
    x3 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp4.to(tl.int1)
    tmp6 = x1
    tmp7 = tl.full([1], 127, tl.int64)
    tmp8 = tmp6 < tmp7
    tmp9 = tmp8.to(tl.int1)
    tmp10 = tmp9 & tmp5
    tmp11 = tl.load(in_ptr0 + (1 + x4), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 30522, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tl.device_assert(((0 <= tl.broadcast_to(tmp15, [XBLOCK])) & (tl.broadcast_to(tmp15, [XBLOCK]) < 30522)) | ~(tmp10), "index out of bounds: 0 <= tl.broadcast_to(tmp15, [XBLOCK]) < 30522")
    tmp17 = tl.load(in_ptr1 + (128*tmp15 + (x0)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp5, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 256, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tmp23.to(tl.int1)
    tmp25 = tl.load(in_ptr0 + (x4), tmp24, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.full([XBLOCK], 30522, tl.int32)
    tmp27 = tmp25 + tmp26
    tmp28 = tmp25 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp25)
    tl.device_assert(((0 <= tl.broadcast_to(tmp29, [XBLOCK])) & (tl.broadcast_to(tmp29, [XBLOCK]) < 30522)) | ~(tmp24), "index out of bounds: 0 <= tl.broadcast_to(tmp29, [XBLOCK]) < 30522")
    tmp31 = tl.load(in_ptr1 + (128*tmp29 + ((-128) + x0)), tmp24, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp0 >= tmp21
    tmp33 = tl.full([1], 384, tl.int64)
    tmp34 = tmp0 < tmp33
    tmp35 = tmp32.to(tl.int1)
    tmp36 = (-1) + x1
    tmp37 = tl.full([1], 0, tl.int64)
    tmp38 = tmp36 >= tmp37
    tmp39 = tmp38.to(tl.int1)
    tmp40 = tmp39 & tmp35
    tmp41 = tl.load(in_ptr0 + ((-1) + x4), tmp40, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.full([XBLOCK], 30522, tl.int32)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp41 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp41)
    tl.device_assert(((0 <= tl.broadcast_to(tmp45, [XBLOCK])) & (tl.broadcast_to(tmp45, [XBLOCK]) < 30522)) | ~(tmp40), "index out of bounds: 0 <= tl.broadcast_to(tmp45, [XBLOCK]) < 30522")
    tmp47 = tl.load(in_ptr1 + (128*tmp45 + ((-256) + x0)), tmp40, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp35, tmp47, tmp48)
    tmp50 = tl.where(tmp23, tmp31, tmp49)
    tmp51 = tl.where(tmp4, tmp19, tmp50)
    tmp52 = tmp51.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp52, None)


def get_args():
    arg_0 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    arg_1 = rand_strided((30522, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 128, 384), (49152, 384, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, arg_2, 6291456,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_constant_pad_nd_embedding_slice_10.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_cat_constant_pad_nd_embedding_slice_10.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.028341248
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
