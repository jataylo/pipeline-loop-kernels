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
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='hip', index=0, multi_processor_count=256, cc='gfx950', major=9, regs_per_multiprocessor=131072, max_threads_per_multi_processor=2048, warp_size=64), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_addmm_mul_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '5E502224A319DB736ED388F470E3117A6892BC105B8AF0DAA4B752DFFD09C80F', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': True, 'min_split_scan_rblock': 256, 'spill_threshold': 32, 'store_cubin': False, 'deterministic': True, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': True, 'is_hip': True, 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'y': 1048576, 'x': 1311744}, 'kernel_num_gb': 0.001311744, 'kernel_flop': 0},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_addmm_mul_transpose_view_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    tl.static_assert(YBLOCK % R0_BLOCK == 0)
    for r in tl.range(0, YBLOCK, R0_BLOCK, num_stages=2):
        lanes = tl.arange(0, R0_BLOCK)
        yindex = yoffset + r + lanes[:, None]
        ymask  = yindex < ynumel
        x2 = xindex
        y0 = (yindex % 512)
        y1 = yindex // 512
        tmp0 = tl.load(in_ptr0 + (x2 + 64*y1 + 256*y0), xmask, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x2 + 64*y1), xmask, eviction_policy='evict_last')
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 0.3535533905932738
        tmp6 = tmp4 * tmp5
        tl.store(out_ptr0 + (y0 + 512*x2 + 32768*y1), tmp6, xmask)
        tl.store(out_ptr1 + (x2 + 64*y1 + 256*y0), tmp6, xmask)


def get_args():
    arg_0 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float16)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 4, 64, 512), (131072, 32768, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((4, 512, 64), (64, 256, 1), device='cuda:0', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3, 2048, 64,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_addmm_mul_transpose_view_7.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused__to_copy_addmm_mul_transpose_view_7.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark_gpu(lambda: call(args), rep=100, warmup=10)
    num_gb = 0.001311744
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
