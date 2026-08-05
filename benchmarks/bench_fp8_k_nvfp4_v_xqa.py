import argparse
import json
import math

import numpy as np
import torch

from flashinfer import xqa
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.testing.utils import bench_gpu_time_with_cudagraph


def _bench(fn) -> float:
    fn()
    torch.cuda.synchronize()
    times = bench_gpu_time_with_cudagraph(
        fn,
        dry_run_time_ms=50,
        repeat_time_ms=300,
        num_iters_within_graph=10,
        cold_l2_cache=False,
    )
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-group-size", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--profile-backend", choices=["xqa", "trtllm-gen"], default=None
    )
    parser.add_argument("--profile-iterations", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    head_dim = 128
    num_q_heads = args.num_kv_heads * args.head_group_size
    pages_per_seq = math.ceil(args.seq_len / args.page_size)
    num_pages = args.batch_size * pages_per_seq

    query = torch.randn(
        args.batch_size,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = (
        torch.randn(
            num_pages,
            args.num_kv_heads,
            args.page_size,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        / 4
    ).to(torch.float8_e4m3fn)
    value_cache = torch.randint(
        0,
        256,
        (
            num_pages,
            args.num_kv_heads,
            args.page_size,
            head_dim // 2,
        ),
        dtype=torch.uint8,
        device=device,
    )
    value_scales = torch.ones(
        num_pages,
        args.num_kv_heads,
        args.page_size,
        head_dim // 16,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    block_tables = torch.arange(
        num_pages, dtype=torch.int32, device=device
    ).reshape(args.batch_size, pages_per_seq)
    seq_lens = torch.full(
        (args.batch_size,), args.seq_len, dtype=torch.int32, device=device
    )
    workspace = torch.empty(256 << 20, dtype=torch.uint8, device=device)
    semaphores = torch.zeros(
        2 * args.batch_size * args.num_kv_heads + 16,
        dtype=torch.uint32,
        device=device,
    )
    output_xqa = torch.empty(
        args.batch_size,
        1,
        num_q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    output_trtllm = torch.empty_like(query)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    def run_trtllm() -> None:
        trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(key_cache, value_cache),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=args.seq_len,
            bmm1_scale=1 / math.sqrt(head_dim),
            bmm2_scale=1.0,
            out=output_trtllm,
            out_dtype=torch.bfloat16,
            kv_layout="HND",
            enable_pdl=True,
            backend="trtllm-gen",
            kv_cache_sf=(None, value_scales),
        )

    if args.profile_backend is not None:
        if args.profile_backend == "trtllm-gen":
            run_profiled = run_trtllm
        else:
            synthetic_sm_count = 2 * args.batch_size * args.num_kv_heads

            def run_profiled() -> None:
                xqa(
                    query.unsqueeze(1),
                    key_cache,
                    value_cache,
                    block_tables,
                    seq_lens.view(torch.uint32).unsqueeze(1),
                    output_xqa,
                    workspace,
                    semaphores,
                    args.num_kv_heads,
                    args.page_size,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    kv_layout="HND",
                    sm_count=synthetic_sm_count,
                    enable_pdl=True,
                    v_sf_cache=value_scales,
                )

        for _ in range(10):
            run_profiled()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(args.profile_iterations):
            run_profiled()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "profile_backend": args.profile_backend,
                    "profile_iterations": args.profile_iterations,
                }
            )
        )
        return

    native_ms = _bench(run_trtllm)
    rows = []
    for splits in args.splits:
        # XQA derives its split count by floor(sm_count / (batch * kv_heads)).
        synthetic_sm_count = splits * args.batch_size * args.num_kv_heads

        def run_xqa() -> None:
            xqa(
                query.unsqueeze(1),
                key_cache,
                value_cache,
                block_tables,
                seq_lens.view(torch.uint32).unsqueeze(1),
                output_xqa,
                workspace,
                semaphores,
                args.num_kv_heads,
                args.page_size,
                k_scale=k_scale,
                v_scale=v_scale,
                kv_layout="HND",
                sm_count=synthetic_sm_count,
                enable_pdl=True,
                v_sf_cache=value_scales,
            )

        xqa_ms = _bench(run_xqa)
        run_trtllm()
        run_xqa()
        torch.cuda.synchronize()
        cosine = torch.nn.functional.cosine_similarity(
            output_xqa.float().flatten(), output_trtllm.float().flatten(), dim=0
        ).item()
        rows.append(
            {
                "splits": splits,
                "synthetic_sm_count": synthetic_sm_count,
                "xqa_ms": xqa_ms,
                "trtllm_gen_ms": native_ms,
                "delta_pct": 100 * (xqa_ms / native_ms - 1),
                "cosine": cosine,
            }
        )

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(device),
                "actual_sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "num_kv_heads": args.num_kv_heads,
                "head_group_size": args.head_group_size,
                "page_size": args.page_size,
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
