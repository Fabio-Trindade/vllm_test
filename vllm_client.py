import argparse
import asyncio
import httpx
import time
import random
from datasets import load_dataset
import traceback
import matplotlib.pyplot as plt
import csv
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json

parser = argparse.ArgumentParser(description="Request data to LLM server")
parser.add_argument("--model", type=str, required=True, help="The model to use for requests")
parser.add_argument("--time", type=int, default=60, help="Total time for requests in seconds")
parser.add_argument("--sleep-time", type=float, default=0.1, help="Sleep time between requests")
parser.add_argument("--using_chunked_prefill", action="store_true", help="Enable chunked prefill")

args = parser.parse_args()

ds = load_dataset("fka/awesome-chatgpt-prompts")
prompts = ds["train"]['prompt']
vllm_server_url = "http://localhost:8000/v1/completions"
seed = 1234
random.seed(seed)

lock = asyncio.Lock()

async def send_data(batched_prompts, client, max_tokens, valid_requests, invalid_requests, total_processed_tokens, inter_token_latencies, ttfts):
    prompts = random.choice(batched_prompts)
    
    request = {
        "model": args.model,
        "prompt": prompts,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True
    }

    previous_times = {prompt_idx: time.time() for prompt_idx in range(len(prompts))}

    try:
        received_prompts = set()

        async with client.stream("POST", vllm_server_url, json=request, timeout=1000) as response:
            async for chunk in response.aiter_lines():
                if  chunk: 
                    if chunk.startswith("data:"):
                        chunk = chunk[len("data:"):].strip()
                    if chunk == "[DONE]":
                        continue
                    data = json.loads(chunk)
                    idx = int(data["choices"][0]["index"])
                    current_time = time.time()

                    if idx not in received_prompts:
                        ttfts.append(current_time - previous_times[idx])
                        received_prompts.add(idx)

                    inter_token_latencies.append(current_time - previous_times[idx])

                    previous_times[idx] = current_time

                    async with lock:
                        total_processed_tokens[0] += 1

        async with lock:
            valid_requests[0] += 1

    except Exception as e:
        print(f"Error during request: {e}")
        traceback.print_exc()
        async with lock:
            invalid_requests[0] += 1


async def main(num_threads, batched_prompts, valid_requests, invalid_requests, total_processed_tokens, ttfts, inter_token_latencies):
    tasks = []
    async with httpx.AsyncClient() as client:
        init_time = time.time()
        elapsed_time = 0

        while elapsed_time < args.time:
            for _ in range(num_threads):
                tasks.append(asyncio.create_task(
                    send_data(batched_prompts, client, max_tokens=50,valid_requests=valid_requests,
                              invalid_requests=invalid_requests, total_processed_tokens=total_processed_tokens,
                              inter_token_latencies=inter_token_latencies, ttfts=ttfts)
                ))
            elapsed_time = time.time() - init_time
            await asyncio.sleep(args.sleep_time)

        await asyncio.gather(*tasks)

def run_experiment(batch_size, writer):
    valid_requests = [0]
    total_processed_tokens = [0]
    invalid_requests = [0]
    inter_token_latencies = []
    ttfts = []

    batched_prompts = [[random.choice(prompts) for _ in range(batch_size)] for _ in range(200)]

    init_time = time.time()
    asyncio.run(main(nt, batched_prompts, valid_requests, invalid_requests, total_processed_tokens, ttfts, inter_token_latencies))
    total_time = time.time() - init_time

    throughput = total_processed_tokens[0] / total_time
    inter_token_latencies = np.array(inter_token_latencies)
    mean_inter_token_latency = inter_token_latencies.mean()
    ITL_median = np.median(inter_token_latencies)
    ttfts = np.array(ttfts)
    mean_ttft = ttfts.mean()
    ttft_median = np.median(ttfts)
    prompts_per_sec = valid_requests[0] * batch_size / total_time
    failure_rate = invalid_requests[0] / (invalid_requests[0] + valid_requests[0])

    assert(len(ttfts) == valid_requests[0]*batch_size)

    writer.add_scalar("Throughput (tokens/sec) vs Batch Size", throughput, batch_size)
    writer.add_scalar("Avg ITL (s/token) vs Batch Size", mean_inter_token_latency, batch_size)
    writer.add_scalar("Median ITL (s/token) vs Batch Size", ITL_median, batch_size)
    writer.add_scalar("Processed Prompts/sec vs Batch Size", prompts_per_sec, batch_size)
    writer.add_scalar("Latency vs Batch Size", total_time, batch_size)
    writer.add_scalar("Mean TTFT (s) vs Batch Size", mean_ttft, batch_size)
    writer.add_scalar("Median TTFT (s) vs Batch Size", ttft_median, batch_size)
    writer.add_scalar("Failure Rate vs Batch Size", failure_rate, batch_size)


writer = SummaryWriter(f"results/tensorboard_logs/{'chunked_prefill' if args.using_chunked_prefill else 'wo_chunked_prefill'}/")
nt = 10
for batch_size in tqdm([2**k for k in range(12)]):
    run_experiment(batch_size, writer)
writer.close()