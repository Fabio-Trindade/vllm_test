import argparse
import asyncio
import httpx
import time
import random
from datasets import load_dataset
import traceback
import numpy as np
import json
from threading import Thread, Lock
import queue
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Request data to LLM server")
parser.add_argument("--model", type=str, required=True, help="The model to use for requests")
parser.add_argument("--time", type=int, default=60, help="Total time for requests in seconds")
parser.add_argument("--sleep-time", type=float, default=0.1, help="Sleep time between requests")
parser.add_argument("--using_chunked_prefill", action="store_true", help="Enable chunked prefill")

args = parser.parse_args()


ds = load_dataset("data-is-better-together/10k_prompts_ranked")
prompts = ds["train"]['prompt']
vllm_server_url = "http://localhost:8000/v1/completions"
random.seed(1234)

metrics_lock = Lock()
async_metrics_lock = asyncio.Lock()  

def init_metrics(id, metrics):
    metrics[id] = {
        "init_time": None,
        "final_time": None,
        "ITLs": [],
        "TTFT": None,
        "num_tokens": None,
    }

def send_data(metrics, q: queue.Queue, prompts, sleep_time, finish):
    id = 0
    while not finish[0]:
        prompt = random.choice(prompts)
        if len(prompt) > 2048:
            continue
        with metrics_lock: 
            init_metrics(id, metrics)
            metrics[id]["num_tokens"] = len(prompt)
            metrics[id]["init_time"] = time.time()
        q.put((id, prompt))
        id += 1
        time.sleep(sleep_time)

async def send_request(args, q: queue.Queue, finish,batch_size, client, max_tokens, metrics):
    while not finish[0] and q.empty():
        continue
    prompts, ids = [], []
    
    async with async_metrics_lock:  
        while len(prompts) < batch_size and not q.empty():
            id, prompt = q.get()
            prompts.append(prompt)
            ids.append(id)

    request = {
        "model": args.model,
        "prompt": prompts,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True
    }

    previous_times = {idx: time.time() for idx in range(len(prompts))}
    received_prompts = set()

    try:
        async with client.stream("POST", vllm_server_url, json=request, timeout=1000) as response:
            async for chunk in response.aiter_lines():
                if chunk:
                    if chunk.startswith("data:"):
                        chunk = chunk[len("data:"):].strip()
                            
                    if chunk == "[DONE]":
                        continue

                    data = json.loads(chunk)
                    if "choices" not in data:
                        print(data)
                    idx = int(data["choices"][0]["index"])
                    true_id = ids[idx]
                    current_time = time.time()

                    async with async_metrics_lock:  
                        if idx not in received_prompts:
                            cur_tft = current_time - previous_times[idx]
                            assert(cur_tft is not None)
                            metrics[true_id]["TTFT"] = cur_tft 
                            received_prompts.add(idx)
                        cur_itl = current_time - previous_times[idx]
                        assert(cur_itl is not None)
                        metrics[true_id]["ITLs"].append(cur_itl)
                        previous_times[idx] = current_time

                        if data["choices"][0]["finish_reason"] != "null":
                            metrics[true_id]["final_time"] = time.time()

    except Exception as e:
        print(f"Error during request: {e}")
        traceback.print_exc()

async def main(q, metrics, prompts, batch_size, max_tokens, nt):
    tasks = []
    finish = [False]

    send_data_thread_list = []
    for _ in range(nt):
        cur_th =  Thread(target=send_data, args=(metrics, q, prompts, args.sleep_time, finish))
        cur_th.start()
        send_data_thread_list.append(cur_th)

    async with httpx.AsyncClient() as client:
        init_time = time.time()

        while time.time() - init_time < args.time:
            tasks.append(asyncio.create_task(
                send_request(args, q, finish,batch_size, client, max_tokens, metrics)
            ))
            await asyncio.sleep(args.sleep_time)


        finish[0] = True
        for th in send_data_thread_list:
            if th.is_alive():  
                th.join() 
        await asyncio.gather(*tasks)

def calc_median_and_percentile(vector):
    array = np.array(vector)
    return np.median(array), np.percentile(array, 99)

def run_experiment(batch_size, writer, nt):
    metrics = {}
    q = queue.Queue()
    
    print(f"Starting experiment with batch size = {batch_size}")
    start_time = time.time()
    asyncio.run(main(q, metrics, prompts, batch_size, 512,nt))
    total_time = time.time() - start_time

    latencies, ttfts, itls = [], [], []
    processed_tokens =  0
    total_processed_prompts = 0
    for metric in metrics.values():
        if metric["TTFT"] == None:
            continue
        total_processed_prompts += 1
        if metric["final_time"] is not None:
            processed_tokens += metric["num_tokens"]
            latencies.append(metric["final_time"] - metric["init_time"])
       
        ttfts.append(metric["TTFT"])
        itls += metric["ITLs"]
    
    print("Total processed prompts:", total_processed_prompts)

    throughput = processed_tokens / total_time
    ttft_median, ttft_99_pct = calc_median_and_percentile(ttfts)
    itl_median, itl_99_pct = calc_median_and_percentile(itls)
    latency_median, latency_99_pct = calc_median_and_percentile(latencies)

    print(f"Results for batch size {batch_size}:")
    print(f"   Throughput: {throughput:.2f} tokens/s")
    print(f"   TTFT: Median {ttft_median:.4f}s | 99th Percentile {ttft_99_pct:.4f}s")
    print(f"   ITL: Median {itl_median:.4f}s | 99th Percentile {itl_99_pct:.4f}s")
    print(f"   Latency: Median {latency_median:.4f}s | 99th Percentile {latency_99_pct:.4f}s")
    print(f"   Unprocessed prompts: {len(metrics)- total_processed_prompts}")
    writer.add_scalar("Throughput", throughput, batch_size)
    writer.add_scalar("TTFT/Median", ttft_median, batch_size)
    writer.add_scalar("TTFT/99th Percentile", ttft_99_pct, batch_size)
    writer.add_scalar("ITL/Median", itl_median, batch_size)
    writer.add_scalar("ITL/99th Percentile", itl_99_pct, batch_size)
    writer.add_scalar("Latency/Median", latency_median, batch_size)
    writer.add_scalar("Latency/99th Percentile", latency_99_pct, batch_size)
    writer.add_scalar("Unprocessed Prompts", len(metrics) - total_processed_prompts, batch_size)


writer = SummaryWriter(f"results/tensorboard_logs/{'chunked_prefill' if args.using_chunked_prefill else 'wo_chunked_prefill'}/")
nt = 16
for batch_size in tqdm([2**k for k in range(12)]):
    run_experiment(batch_size, writer, nt)
writer.close()
