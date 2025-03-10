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
parser = argparse.ArgumentParser(description="Request data to LLM server")
parser.add_argument("--model", type=str, required=True, help="The model to use for requests")
parser.add_argument("--time", type=int, default=60, help="Total time for requests in seconds")
parser.add_argument("--sleep-time", type=float, default=0.1, help="Sleep time between requests")


args = parser.parse_args()
ds = load_dataset("fka/awesome-chatgpt-prompts")
prompts = ds["train"]['prompt']
vllm_server_url = "http://localhost:8000/v1/completions"  
seed = 1234

lock = asyncio.Lock()

async def send_data(prompts, client, max_tokens):
    global total_requests, total_processed_tokens, inter_token_latencies

    prompt = random.choice(prompts)
    request = {
        "model": args.model,
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True 
    }

    try:
        response = await client.post(vllm_server_url, json=request, timeout=1000)
        
        if response.status_code == 200:
            first_token_time = None
            previous_time = None

            async for chunk in response.aiter_lines():
                if chunk:
                    current_time = time.time()

                    if first_token_time is None:
                        first_token_time = current_time
                    else:
                        inter_token_latency = current_time - previous_time
                        inter_token_latencies.append(inter_token_latency)
                    previous_time = current_time
                    async with lock:
                        total_processed_tokens += 1
            async with lock:
                total_requests += 1

    except httpx.RequestError as e:
        print(f"Request error occurred: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

async def main(num_threads):
    global total_requests, total_processed_tokens, batched_prompts
    tasks = []
    async with httpx.AsyncClient() as client:
        init_time = time.time()
        elapsed_time = 0

        while elapsed_time < args.time:
            for _ in range(num_threads):
                tasks.append(asyncio.create_task(send_data(batched_prompts, client, max_tokens=50)))
            elapsed_time = time.time() - init_time
            await asyncio.sleep(args.sleep_time)  

        await asyncio.gather(*tasks)

nt = 5
for use_chunked_prefill in [True,False]:
    writer = SummaryWriter(f"results/tensorboard_logs/{"chunked_prefill" if use_chunked_prefill else "wo_chunked_prefill"}/")
    for batch_size in tqdm([2**k for k in range(12)]):
        total_requests = 0
        total_processed_tokens = 0
        inter_token_latencies = [] 
        batched_prompts = [[prompts[random.randint(0, len(prompts) - 1)] for i in range(batch_size)] for j in range(50)]

        init_time = time.time()
        asyncio.run(main(nt))

        total_time = time.time() - init_time
        throughput = total_processed_tokens / total_time 
        inter_token_latencies = np.array(inter_token_latencies)
        mean_inter_token_latency = inter_token_latencies.mean()
        ITL_median = np.median(inter_token_latencies)
        prompts_per_sec = total_requests*batch_size / total_time
        latency = total_time 

        # print()
        # print(f"Threads: {nt}")
        # print("Throughput (tokens/sec):", throughput)
        # print("Average Inter-token latency (sec/token):", mean_inter_token_latency)
        # print("ITL median:", ITL_median)

        writer.add_scalar("Throughput (tokens/sec) vs Batch Size", throughput, batch_size)
        writer.add_scalar("Avg ITL (s/token) vs Batch Size", mean_inter_token_latency, batch_size)
        writer.add_scalar("Median ITL (s/token) vs Batch Size", ITL_median, batch_size)
        writer.add_scalar("Prompts/sec vs Batch Size", prompts_per_sec, batch_size)
        writer.add_scalar("Latency vs Batch Size", total_time, batch_size)
    writer.close()

