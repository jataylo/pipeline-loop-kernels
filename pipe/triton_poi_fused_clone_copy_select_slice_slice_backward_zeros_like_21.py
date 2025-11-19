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
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_select_slice_slice_backward_zeros_like_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'is_hip': True, 'tiling_scores': {'x': 100859904}, 'kernel_num_gb': 0.075644928, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_copy_select_slice_slice_backward_zeros_like_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12607488
    xoffset = tl.program_id(0) * XBLOCK
    tl.static_assert(XBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, XBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        xindex = xoffset + r + lanes[:]
        xmask  = xindex < xnumel
        x0 = (xindex % 513)
        x1 = ((xindex // 513) % 512)
        x2 = xindex // 262656
        x4 = (xindex % 262656)
        x5 = xindex
        tmp0 = x1
        tmp1 = tl.full([1], 256, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tmp2.to(tl.int1)
        tmp4 = x0
        tmp5 = tl.full([1], 257, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tmp6.to(tl.int1)
        tmp8 = tmp7 & tmp3
        tmp9 = tl.full([1], 3, tl.int64)
        tmp10 = tl.full([1], 1, tl.int64)
        tmp11 = tmp9 >= tmp10
        tmp12 = tmp11.to(tl.int1)
        tmp13 = tmp12 & tmp8
        tmp14 = 256 + x0
        tmp15 = tl.full([1], 256, tl.int64)
        tmp16 = tmp14 < tmp15
        tmp17 = tmp16.to(tl.int1)
        tmp18 = tmp17 & tmp13
        tmp19 = 0.0
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp18, tmp19, tmp20)
        tmp22 = tl.full([1], 3, tl.int32)
        tmp23 = tl.full([1], 0, tl.int32)
        tmp24 = tmp22 == tmp23
        tmp25 = (-256) + x1
        tmp26 = tl.full([1], 1, tl.int64)
        tmp27 = tmp25 >= tmp26
        tmp28 = tmp27.to(tl.int1)
        tmp29 = tmp28 & tmp13
        tmp30 = 256 + x0
        tmp31 = tl.full([1], 1, tl.int64)
        tmp32 = tmp30 >= tmp31
        tmp33 = tl.full([1], 256, tl.int64)
        tmp34 = tmp30 < tmp33
        tmp35 = tmp32 & tmp34
        tmp36 = tmp35.to(tl.int1)
        tmp37 = tmp36 & tmp29
        tmp38 = 0.0
        tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
        tmp40 = tl.where(tmp37, tmp38, tmp39)
        tmp41 = tl.load(in_ptr0 + ((-131072) + x4 + 525312*x2), tmp29, other=0.0).to(tl.float32)
        tmp42 = tl.where(tmp35, tmp40, tmp41)
        tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
        tmp44 = tl.where(tmp29, tmp42, tmp43)
        tmp45 = tl.load(in_ptr0 + ((-131072) + x4 + 525312*x2), tmp13, other=0.0).to(tl.float32)
        tmp46 = tl.where(tmp27, tmp44, tmp45)
        tmp47 = tl.load(in_ptr0 + (262912 + x4 + 525312*x2), tmp13, other=0.0).to(tl.float32)
        tmp48 = tl.where(tmp24, tmp46, tmp47)
        tmp49 = tl.where(tmp16, tmp21, tmp48)
        tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
        tmp51 = tl.where(tmp13, tmp49, tmp50)
        tmp52 = tl.full([1], 3, tl.int32)
        tmp53 = tl.full([1], 0, tl.int32)
        tmp54 = tmp52 == tmp53
        tmp55 = (-256) + x1
        tmp56 = tmp55 >= tmp10
        tmp57 = tmp56.to(tl.int1)
        tmp58 = tmp57 & tmp8
        tmp59 = 256 + x0
        tmp60 = tl.full([1], 1, tl.int64)
        tmp61 = tmp59 >= tmp60
        tmp62 = tl.full([1], 256, tl.int64)
        tmp63 = tmp59 < tmp62
        tmp64 = tmp61 & tmp63
        tmp65 = tmp64.to(tl.int1)
        tmp66 = tmp65 & tmp58
        tmp67 = 0.0
        tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
        tmp69 = tl.where(tmp66, tmp67, tmp68)
        tmp70 = tl.load(in_ptr0 + ((-131072) + x4 + 525312*x2), tmp58, other=0.0).to(tl.float32)
        tmp71 = tl.where(tmp64, tmp69, tmp70)
        tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
        tmp73 = tl.where(tmp58, tmp71, tmp72)
        tmp74 = tl.load(in_ptr0 + ((-131072) + x4 + 525312*x2), tmp8, other=0.0).to(tl.float32)
        tmp75 = tl.where(tmp56, tmp73, tmp74)
        tmp76 = tl.load(in_ptr0 + (262912 + x4 + 525312*x2), tmp8, other=0.0).to(tl.float32)
        tmp77 = tl.where(tmp54, tmp75, tmp76)
        tmp78 = tl.where(tmp11, tmp51, tmp77)
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp8, tmp78, tmp79)
        tmp81 = 0.0
        tmp82 = tl.where(tmp6, tmp80, tmp81)
        tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
        tmp84 = tl.where(tmp3, tmp82, tmp83)
        tmp85 = 0.0
        tmp86 = tl.where(tmp2, tmp84, tmp85)
        tl.store(out_ptr0 + (x5), tmp86, None)


def get_args():
    arg_0 = rand_strided((48, 4, 256, 513), (525312, 131328, 513, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((48, 512, 513), (262656, 513, 1), device='cuda:0', dtype=torch.float16)
    return arg_0, arg_1, 12607488,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_copy_select_slice_slice_backward_zeros_like_21.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_clone_copy_select_slice_slice_backward_zeros_like_21.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.075644928
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
