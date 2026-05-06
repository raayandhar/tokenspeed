"""Benchmark runner for agentic multi-turn workloads.

Loads a dataset built by build_dataset.py and sends multi-turn
conversations to a TokenSpeed server via /v1/chat/completions.

Each client processes one or more conversations sequentially. Within each
conversation, turns are sent one at a time: the server response from turn N
is inserted as an assistant message before sending turn N+1's delta messages.

Metrics are aligned with TensorRT-LLM benchmark_serving.py format.

Usage:
    python bench.py --dataset-path agentic_dataset.json --num-clients 32
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, List

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark agentic multi-turn workloads against a TokenSpeed server."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to agentic dataset JSON (built by build_dataset.py)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum turns for each conversation (default: 1)",
    )
    parser.add_argument(
        "--num-requests-per-client",
        type=int,
        default=1,
        help="Number of conversations per client (default: 1)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=32,
        help="Number of concurrent clients (default: 32)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--flush-cache",
        action="store_true",
        default=False,
        help="Flush server cache before running benchmark",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="performance_metrics.jsonl",
        help="Output metrics file (default: performance_metrics.jsonl)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Tag for this run in the log file",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Rotate conversations by offset to avoid cache hits across runs (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    return parser.parse_args()


async def send_chat_completion(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    output_length: int,
) -> Dict:
    """Send a streaming chat completion request. Measures TTFT, TPOT, E2EL."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": output_length,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    st = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}",
                }

            ttft = None
            content_parts = []
            usage = {}
            num_content_chunks = 0

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break

                chunk = json.loads(data_str)

                # Final chunk with usage (empty choices)
                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is not None:
                    if ttft is None:
                        ttft = time.perf_counter() - st
                    content_parts.append(content)
                    num_content_chunks += 1

            e2el = time.perf_counter() - st

            prompt_tokens_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0)

            # TPOT: (e2el - ttft) / (completion_tokens - 1), aligned with TRT-LLM
            tpot = None
            if ttft is not None and completion_tokens > 1:
                tpot = (e2el - ttft) / (completion_tokens - 1)

            if num_content_chunks > 0:
                avg_decoded_tokens_per_iter = completion_tokens / num_content_chunks
            else:
                avg_decoded_tokens_per_iter = 0.0

            return {
                "success": True,
                "content": "".join(content_parts),
                "e2el": e2el,
                "ttft": ttft or e2el,
                "tpot": tpot,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "cached_tokens": cached_tokens,
                "completion_tokens": completion_tokens,
                "avg_decoded_tokens_per_iter": avg_decoded_tokens_per_iter,
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


METRIC_KEYS = [
    "ttft",
    "tpot",
    "e2el",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "avg_decoded_tokens_per_iter",
]


def new_metric_lists():
    return {k: [] for k in METRIC_KEYS}


async def run_client(
    client_id: int,
    conv_queue: asyncio.Queue,
    url: str,
    model: str,
    output_length: int,
    session: aiohttp.ClientSession,
    metrics: Dict,
    pbar: tqdm,
):
    """Run a single client that pulls conversations from a shared queue."""
    while not conv_queue.empty():
        try:
            conv = conv_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        messages = []
        num_turns = len(conv)

        for turn_idx, turn in enumerate(conv):
            messages.extend(turn["messages"])

            result = await send_chat_completion(
                session, url, model, messages, output_length
            )

            if not result["success"]:
                print(f"Client {client_id} turn {turn_idx} failed: {result['error']}")
                break

            # Record metrics
            turn_key = f"turn_{turn_idx}"
            for key in METRIC_KEYS:
                val = result[key]
                if val is not None:
                    metrics["all"][key].append(val)

            if turn_key not in metrics["per_turn"]:
                metrics["per_turn"][turn_key] = new_metric_lists()
            for key in METRIC_KEYS:
                val = result[key]
                if val is not None:
                    metrics["per_turn"][turn_key][key].append(val)

            pbar.update(1)

            # Insert server response for next turn
            if turn_idx < num_turns - 1:
                messages.append({"role": "assistant", "content": result["content"]})


def process_one_metric(values, name, header, unit_suffix=" (ms)", scale=1000.0):
    """Print stats for one metric, aligned with TRT-LLM format."""
    if not values:
        return {}
    arr = np.array(values) * scale
    stats = {
        f"mean_{name}": float(np.mean(arr)),
        f"p50_{name}": float(np.median(arr)),
        f"p90_{name}": float(np.percentile(arr, 90)),
    }
    print("{s:{c}^{n}}".format(s=f" {header} ", n=50, c="-"))
    print(
        "{:<40} {:<10.2f}".format(f"Mean {name}{unit_suffix}:", stats[f"mean_{name}"])
    )
    print("{:<40} {:<10.2f}".format(f"P50 {name}{unit_suffix}:", stats[f"p50_{name}"]))
    print("{:<40} {:<10.2f}".format(f"P90 {name}{unit_suffix}:", stats[f"p90_{name}"]))
    return stats


def compute_and_print_results(metrics, duration, num_requests):
    """Compute and print results in TRT-LLM format. Returns results dict."""
    m = metrics["all"]
    total_input = sum(m["prompt_tokens"])
    total_output = sum(m["completion_tokens"])
    completed = len(m["ttft"])

    # User throughput: mean of per-request (output_tokens / e2el)
    user_tput = []
    for ct, e2el in zip(m["completion_tokens"], m["e2el"]):
        if e2el > 0:
            user_tput.append(ct / e2el)

    # === Summary ===
    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Total requests:", completed))
    print("{:<40} {:<10}".format("Total conversations:", num_requests))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", duration))
    print("{:<40} {:<10}".format("Total input tokens:", total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", completed / duration if duration > 0 else 0
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):",
            total_output / duration if duration > 0 else 0,
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):",
            (total_input + total_output) / duration if duration > 0 else 0,
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "User throughput (tok/s):", np.mean(user_tput) if user_tput else 0
        )
    )
    print(
        "{:<40} {:<10.4f}".format(
            "Cache hit rate:",
            sum(m["cached_tokens"]) / total_input if total_input > 0 else 0,
        )
    )

    # === Per-metric sections ===
    all_stats = {}
    all_stats.update(
        process_one_metric(
            m["avg_decoded_tokens_per_iter"],
            "Avg Decoded Tokens per Iter",
            "Avg Decoded Tokens per Iter",
            unit_suffix="",
            scale=1.0,
        )
    )
    all_stats.update(process_one_metric(m["ttft"], "TTFT", "Time to First Token"))
    all_stats.update(
        process_one_metric(m["tpot"], "TPOT", "Time per Output Token (excl. 1st token)")
    )
    all_stats.update(process_one_metric(m["e2el"], "E2EL", "End-to-end Latency"))

    # === Per-turn table ===
    print("\n{s:{c}^{n}}".format(s=" Per-turn Metrics ", n=50, c="-"))
    header = (
        f"  {'Turn':<6} {'Count':<7} {'TTFT':<10} {'TPOT':<10} "
        f"{'E2EL':<10} {'Prompt':<10} {'Output':<10} "
        f"{'Cached':<10} {'Cache%':<10}"
    )
    print(header)
    per_turn = {}
    for turn_key, tm in sorted(metrics["per_turn"].items()):
        turn_num = turn_key.split("_")[1]
        avg_ttft = np.mean(tm["ttft"]) * 1000 if tm["ttft"] else 0
        avg_tpot = np.mean(tm["tpot"]) * 1000 if tm["tpot"] else 0
        avg_e2el = np.mean(tm["e2el"]) * 1000 if tm["e2el"] else 0
        avg_prompt = np.mean(tm["prompt_tokens"]) if tm["prompt_tokens"] else 0
        avg_output = np.mean(tm["completion_tokens"]) if tm["completion_tokens"] else 0
        avg_cached = np.mean(tm["cached_tokens"]) if tm["cached_tokens"] else 0
        cache_hit = (
            sum(tm["cached_tokens"]) / sum(tm["prompt_tokens"])
            if sum(tm["prompt_tokens"]) > 0
            else 0
        )
        print(
            f"  {turn_num:<6} {len(tm['ttft']):<7} {avg_ttft:<10.2f} "
            f"{avg_tpot:<10.2f} {avg_e2el:<10.2f} {avg_prompt:<10.0f} "
            f"{avg_output:<10.0f} {avg_cached:<10.0f} {cache_hit:<10.4f}"
        )
        per_turn[turn_key] = {
            "count": len(tm["ttft"]),
            "mean_ttft_ms": round(avg_ttft, 2),
            "mean_tpot_ms": round(avg_tpot, 2),
            "mean_e2el_ms": round(avg_e2el, 2),
            "mean_prompt_tokens": round(avg_prompt, 0),
            "mean_completion_tokens": round(avg_output, 0),
            "mean_cached_tokens": round(avg_cached, 0),
            "cache_hit_rate": round(cache_hit, 4),
        }

    print("=" * 50)

    # Build results dict for logging
    results = {
        "summary": {
            "total_requests": completed,
            "num_requests": num_requests,
            "duration_s": round(duration, 2),
            "total_input_tokens": total_input,
            "total_generated_tokens": total_output,
            "request_throughput": round(completed / duration, 2) if duration > 0 else 0,
            "output_token_throughput": (
                round(total_output / duration, 2) if duration > 0 else 0
            ),
            "total_token_throughput": (
                round((total_input + total_output) / duration, 2) if duration > 0 else 0
            ),
            "user_throughput": round(np.mean(user_tput), 2) if user_tput else 0,
            "cache_hit_rate": round(
                sum(m["cached_tokens"]) / total_input if total_input > 0 else 0, 4
            ),
            **all_stats,
        },
        "per_turn": per_turn,
    }
    return results


def log_to_jsonl(data, file_path, tag=""):
    """Append results to a JSONL file."""
    entry = {"timestamp": datetime.now().isoformat(), "tag": tag, **data}
    with open(file_path, "w") as f:
        f.write(json.dumps(entry) + "\n")


async def run_benchmark(args):
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path) as f:
        dataset = json.load(f)

    metadata = dataset["metadata"]
    conversations = dataset["conversations"]
    if args.max_turns is not None:
        conversations = [conv[: args.max_turns] for conv in conversations]
    model = metadata["model_path"]
    output_length = metadata["output_length"]
    num_turns = len(conversations[0])

    print(f"  Model: {model}")
    print(f"  Conversations: {len(conversations)}")
    print(f"  Turns per conversation: {num_turns}")
    print(f"  Output length: {output_length}")

    # Determine num_requests
    num_clients = args.num_clients
    num_requests_per_client = args.num_requests_per_client
    num_requests = num_clients * num_requests_per_client
    if num_requests > len(conversations):
        raise ValueError(
            f"Error: need {num_requests} conversations "
            f"({num_requests_per_client} x {num_clients} clients) "
            f"but dataset only has {len(conversations)}"
        )

    # Shuffle then rotate — offset guarantees different conversations across runs
    random.seed(args.seed)
    random.shuffle(conversations)
    offset = args.offset % len(conversations)
    conversations = conversations[offset:] + conversations[:offset]
    selected = conversations[:num_requests]

    total_http_requests = num_requests * num_turns
    print(
        f"\nRunning {num_requests} conversations "
        f"({num_requests_per_client} x {num_clients} clients)"
    )
    print(f"Total HTTP requests: {total_http_requests}")

    # Flush cache if requested
    if args.flush_cache:
        flush_url = f"http://{args.host}:{args.port}/flush_cache"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    flush_url, timeout=aiohttp.ClientTimeout(total=30)
                ):
                    print("Cache flushed successfully")
        except Exception as e:
            print(f"Warning: Failed to flush cache: {e}")

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    metrics = {
        "all": new_metric_lists(),
        "per_turn": {},
    }

    pbar = tqdm(total=total_http_requests, desc="Requests")

    # Shared queue — clients pull conversations on demand
    conv_queue = asyncio.Queue()
    for conv in selected:
        conv_queue.put_nowait(conv)

    start_time = time.perf_counter()
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        tasks = [
            run_client(i, conv_queue, url, model, output_length, session, metrics, pbar)
            for i in range(num_clients)
        ]
        await asyncio.gather(*tasks)
    duration = time.perf_counter() - start_time
    pbar.close()

    # Compute and print results
    results = compute_and_print_results(metrics, duration, num_requests)
    results["metadata"] = {
        "dataset_path": args.dataset_path,
        "num_requests": num_requests,
        "num_requests_per_client": num_requests_per_client,
        "num_clients": num_clients,
        "num_turns": num_turns,
        "output_length": output_length,
    }

    log_to_jsonl(results, args.log_file, tag=args.tag)
    print(f"\nResults logged to {args.log_file}")

    return results


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
