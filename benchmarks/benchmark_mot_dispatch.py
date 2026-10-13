"""Matched MoT dispatch diagnostic; excludes router and block projection cost."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.suite import collect_environment, resolve_device, summarize_latency
from ultralytics.nn.modules.mot import MoTBlock


def routing_inputs(block, x, pattern):
    """Create Top-1 routes with known sample coverage, or use the native router."""
    batch, _, height, width = x.shape
    if pattern == "native":
        return block.router(x)
    indices = torch.zeros((batch, 1, height, width), device=x.device, dtype=torch.long)
    if pattern == "spatial_union":
        indices[:] = torch.arange(width, device=x.device).remainder(block.num_experts).view(1, 1, 1, width)
    elif pattern == "sample_split":
        indices[:] = torch.arange(batch, device=x.device).remainder(block.num_experts).view(batch, 1, 1, 1)
    elif pattern != "single":
        raise ValueError(f"unknown routing pattern: {pattern}")
    weights = x.new_zeros((batch, block.num_experts, height, width))
    weights.scatter_(1, indices, 1.0)
    return weights, indices


def dense_reference(block, x, weights):
    """Execute every expert with the same masked weights as sparse dispatch."""
    result = torch.zeros_like(x)
    for expert, weight in zip(block.experts, weights.unbind(1)):
        result = result + expert(x) * weight.unsqueeze(1)
    return result


def synchronize(device):
    """Complete accelerator work before reading host timers."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(call, device, warmup, steps, batch):
    """Measure synchronized dispatch forwards without autograd."""
    for _ in range(warmup):
        call()
    synchronize(device)
    times = []
    for _ in range(steps):
        synchronize(device)
        start = time.perf_counter()
        call()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return summarize_latency(times, batch)


def observe_execution(block, call):
    """Count actual calls and sample-expert pairs using untimed hooks."""
    counts = {"expert_calls": 0, "sample_expert_pairs": 0}

    def observe(module, inputs):
        counts["expert_calls"] += 1
        counts["sample_expert_pairs"] += int(inputs[0].shape[0])

    hooks = [expert.register_forward_pre_hook(observe) for expert in block.experts]
    try:
        output = call()
    finally:
        for hook in hooks:
            hook.remove()
    return output, counts


@torch.inference_mode()
def run_case(block, x, pattern, warmup=3, steps=20):
    """Check matched outputs, then report logical sparsity and actual execution."""
    if block.training:
        raise ValueError("dispatch diagnostic requires an eval block")
    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")
    weights, indices = routing_inputs(block, x, pattern)
    calls = {
        "dense_masked_reference": lambda: dense_reference(block, x, weights),
        "sample_sparse": lambda: block._blend_experts(x, weights, indices),
    }
    outputs, observations = {}, {}
    for mode, call in calls.items():
        outputs[mode], observations[mode] = observe_execution(block, call)
    torch.testing.assert_close(outputs["sample_sparse"], outputs["dense_masked_reference"], rtol=1e-4, atol=1e-5)
    dense_pairs = x.shape[0] * block.num_experts
    return {
        "pattern": pattern,
        "shape": list(x.shape),
        "max_abs_error": float((outputs["sample_sparse"] - outputs["dense_masked_reference"]).abs().max()),
        "token_mask_sparsity": 1.0 - float((weights != 0).float().mean()),
        "sample_expert_pair_reduction": 1.0 - observations["sample_sparse"]["sample_expert_pairs"] / dense_pairs,
        "modes": {
            mode: {**observations[mode], **measure(call, x.device, warmup, steps, x.shape[0])}
            for mode, call in calls.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.channels, args.threads) < 1 or args.size < 3 or args.warmup < 0:
        parser.error("steps/batch/channels/threads must be positive, size >= 3, warmup >= 0")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    block = MoTBlock(args.channels, num_heads=4, top_k=1).eval().to(device)
    x = torch.randn(args.batch, args.channels, args.size, args.size, device=device)
    with torch.inference_mode():
        report = {
            "schema_version": 1,
            "scope": "synthetic expert dispatch only; excludes router, projection, AP and end-to-end latency",
            "environment": collect_environment(),
            "settings": {**vars(args), "device": str(device)},
            "cases": [
                run_case(block, x, pattern, args.warmup, args.steps)
                for pattern in ("single", "sample_split", "spatial_union", "native")
            ],
        }
    text = json.dumps(report, indent=2, default=str, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
