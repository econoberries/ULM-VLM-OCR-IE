"""
Computational metrics for the minicpm runs

Per record: inference_seconds, peak_allocated_gpu_bytes, output_tokens             
Per run (<output>.run_metadata.json): model_load_seconds, started_utc, torch_version, cuda_device_names.

uv run python src/vlm/minicpm/efficiency.py --predictions P.json --metadata P.run_metadata.json

References
1. QWEN-RUN, Qwen3.5 inference script: src/vlm/qwen35/run.py, branch feature/qwen35-english-kie, commit 969c420
2. QWEN-SUM, Qwen3.5 efficiency summary: summaries/qwen35_english_kie_400_efficiency_summary.json in qwen35_helix_evidence_v1.zip
"""

import json
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from statistics import mean, median

import torch


# QWEN-RUN, lines 69-72
def _synchronize_cuda():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


# QWEN-RUN, lines 87-90
def start_sample_timer():
    """Reset GPU peak memory and start the clock."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    _synchronize_cuda()
    return time.perf_counter()


# QWEN-RUN, lines 75-78, 97-98, 197
def stop_sample_timer(start):
    """Return (inference_seconds, peak_allocated_gpu_bytes); empty GPU list on CPU."""
    _synchronize_cuda()
    seconds = round(time.perf_counter() - start, 4)
    peak = [int(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
    return seconds, peak


# QWEN-RUN, lines 180-182
def environment_info():
    """Run-level environment fields."""
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def _percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q / 100
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def _is_strict_json_object(text):
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False


# QWEN-SUM
def summarize_efficiency(rows, metadata=None):
    """Efficiency summary with the Qwen3.5 field names."""
    metadata = metadata or {}
    latencies = [r["inference_seconds"] for r in rows if isinstance(r.get("inference_seconds"), (int, float))]
    peaks_gib = [max(r["peak_allocated_gpu_bytes"]) / 2**30 for r in rows if r.get("peak_allocated_gpu_bytes")]
    tokens = [r["output_tokens"] for r in rows if isinstance(r.get("output_tokens"), int)]
    predicts = [r["predict"] for r in rows if isinstance(r.get("predict"), str)]
    valid_json = sum(_is_strict_json_object(p) for p in predicts)

    def stat(fn, values):
        return fn(values) if values else None

    return {
        "samples": len(rows),
        # Qwen: rows with error
        # MiniCPM: metadata failed_ids
        "inference_failures": sum(bool(r.get("error")) for r in rows) + len(metadata.get("failed_ids", [])),
        "mean_latency_seconds": stat(mean, latencies),
        "median_latency_seconds": stat(median, latencies),
        "p95_latency_seconds": stat(lambda v: _percentile(v, 95), latencies),
        "min_latency_seconds": stat(min, latencies),
        "max_latency_seconds": stat(max, latencies),
        "total_inference_seconds": stat(sum, latencies),
        "mean_output_tokens": stat(mean, tokens),
        "mean_peak_allocated_gpu_gib": stat(mean, peaks_gib),
        "maximum_peak_allocated_gpu_gib": stat(max, peaks_gib),
        "valid_json_documents": valid_json,
        "strict_json_validity_rate": valid_json / len(rows) if rows else None,
        "code_fenced_documents": sum("```" in p for p in predicts),
        "model_loading_seconds": metadata.get("model_load_seconds"),
        "gpu_names": metadata.get("cuda_device_names"),
    }


def main():
    parser = ArgumentParser(description="Summarize computational metrics of a prediction file.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON list (raw or scored)")
    parser.add_argument("--metadata", help="Matching .run_metadata.json (model load time, GPU names)")
    parser.add_argument("--output", help="Optional path to save the summary JSON")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        rows = json.load(f)
    metadata = None
    if args.metadata:
        with open(args.metadata, encoding="utf-8") as f:
            metadata = json.load(f)
    summary = summarize_efficiency(rows, metadata)
    print(json.dumps(summary, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
