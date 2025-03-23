import argparse
import asyncio
import httpx
import time
import random
from datasets import load_dataset
import traceback
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Request data to LLM server")
parser.add_argument("--model", type=str, required=True, help="The model to use for requests")
parser.add_argument("--time", type=int, default=60, help="Total time for requests in seconds")
# parser.add_argument("--sleep-time", type=float, default=0.1, help="Sleep time between requests")
parser.add_argument("--sleep-time-queue", type=float, default=0.1, help="Sleep time between requests")
parser.add_argument("--sleep-time-request", type=float, default=0.1, help="Sleep time between requests")
parser.add_argument("--using_chunked_prefill",action="store_true", help="Enable chunked prefill")
parser.add_argument("--model-seq-len",  type=int, help="")

args = parser.parse_args()

ds = load_dataset("data-is-better-together/10k_prompts_ranked")
prompts = ds["train"]['prompt']
vllm_server_url = "http://localhost:8000/v1/completions"
random.seed(1234)

lock = asyncio.Lock()

def init_metrics(id, metrics):
    assert id not in metrics
    metrics[id] = {
        "init_time": None,
        "final_time": None,
        "req_init_time": None,
        "prefill_final_time":None,
        "ITLs": [],
        "TTFT": None,
        "num_input_tokens": None,
    }


async def send_data_to_queue(metrics, q: asyncio.Queue, prompts, sleep_time, model_seq_len, finish):
    while not finish[0]:
        prompt = random.choice(prompts)
        if len(prompt) > model_seq_len:
            continue
        async with lock:
            req_id = len(metrics)
            init_metrics(req_id, metrics)
            metrics[req_id]["num_input_tokens"] = len(prompt)
            metrics[req_id]["init_time"] = time.time()
        await q.put((req_id, prompt))
        await asyncio.sleep(sleep_time)

async def send_request(args, q: asyncio.Queue, finish, batch_size, client, max_tokens, metrics):
    while not finish[0] and q.empty():
        await asyncio.sleep(0.1)
    prompts, ids = [], []
    
    async with lock:
        while len(prompts) < batch_size and not q.empty():
            id, prompt = await q.get()
            prompts.append(prompt)
            ids.append(id)
            metrics[id]["req_init_time"] = time.time()

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
        async with client.stream("POST", vllm_server_url, json=request, timeout=300) as response:
            async for chunk in response.aiter_lines():
                if chunk:
                    if chunk.startswith("data:"):
                        chunk = chunk[len("data:"):].strip()
                            
                    if chunk == "[DONE]":
                        continue

                    data = json.loads(chunk)
                    
                    idx = int(data["choices"][0]["index"])
                    true_id = ids[idx]
                    current_time = time.time()

                    async with lock:
                        if idx not in received_prompts:
                            cur_tft = current_time - previous_times[idx]
                            metrics[true_id]["TTFT"] = cur_tft 
                            metrics[true_id]["prefill_final_time"] = time.time()
                            received_prompts.add(idx)
                        else:
                            cur_itl = current_time - previous_times[idx]
                            metrics[true_id]["ITLs"].append(cur_itl)
                        
                        previous_times[idx] = current_time

                        if data["choices"][0]["finish_reason"] != "null":
                            metrics[true_id]["final_time"] = time.time()

    except Exception as e:
        print(f"Error during request: {e}")
        # traceback.print_exc()

async def continuous_request(q, metrics, prompts, batch_size, max_tokens, nt):
    tasks = []
    finish = [False]

    send_data_tasks = [asyncio.create_task(send_data_to_queue(metrics, q, prompts, args.sleep_time_queue, args.model_seq_len, finish)) for _ in range(nt)]

    async with httpx.AsyncClient() as client:
        init_time = time.time()
        while time.time() - init_time < args.time:
            tasks.append(asyncio.create_task(
                send_request(args, q, finish, batch_size, client, max_tokens, metrics)
            ))
            await asyncio.sleep(args.sleep_time_request)

        finish[0] = True
        await asyncio.gather(*send_data_tasks)
        await asyncio.gather(*tasks)

def calc_median_and_percentile(vector):
    array = np.array(vector)
    return np.median(array), np.percentile(array, 99)

async def run_experiment(batch_size, writer, nt):
    metrics = {}
    q = asyncio.Queue()
    
    print(f"Starting experiment with batch size = {batch_size}")
    start_time = time.time()

    task1 = asyncio.create_task(continuous_request(q, metrics, prompts, batch_size, 512, nt))
    task2 = asyncio.create_task(continuous_request(q, metrics, prompts, batch_size, 512, nt))

    await asyncio.gather(task1, task2)
    total_time = time.time() - start_time

    latencies, ttfts, itls, decode_times, time_in_queue = [], [], [], [], []
    processed_tokens =  0
    total_processed_prompts = 0
    for metric in metrics.values():
        if metric["TTFT"] is None:
            continue
        total_processed_prompts += 1
        ttfts.append(metric["TTFT"])
        itls += metric["ITLs"]
        if metric["final_time"] is not None:
            processed_tokens += metric["num_input_tokens"] + len(metric["ITLs"]) + 1
            latencies.append(metric["final_time"] - metric["init_time"])
            decode_times.append(metric["final_time"] - metric["prefill_final_time"])
            time_in_queue.append(metric["req_init_time"] - metric["init_time"])
       

    throughput = processed_tokens / total_time
    ttft_median, ttft_99_pct = calc_median_and_percentile(ttfts)
    itl_median, itl_99_pct = calc_median_and_percentile(itls)
    latency_median, latency_99_pct = calc_median_and_percentile(latencies)
    decode_median, decode_99_pct = calc_median_and_percentile(decode_times)
    time_in_queue_median, time_in_queue_99_pct = calc_median_and_percentile(time_in_queue)


    prompts_per_second = total_processed_prompts / total_time
    unprocessed_server_prompts = len(metrics) - total_processed_prompts - q.qsize()

    print(f"Results for batch size {batch_size}:")

    print("Total processed prompts:", total_processed_prompts)
    print("Total processed tokens:", processed_tokens)
    print("Total metrics:", len(metrics))
    print("Total time:", total_time)
    print(f"   Throughput: {throughput:.2f} tokens/s")
    print(f"   TTFT: Median {ttft_median:.4f}s | 99th Percentile {ttft_99_pct:.4f}s")
    print(f"   ITL: Median {itl_median:.4f}s | 99th Percentile {itl_99_pct:.4f}s")
    print(f"   Latency: Median {latency_median:.4f}s | 99th Percentile {latency_99_pct:.4f}s")
    print(f"   Decode Time: Median {decode_median:.4f}s | 99th Percentile {decode_99_pct:.4f}s")
    print(f"   Time in Queue: Median {time_in_queue_median:.4f}s | 99th Percentile {time_in_queue_99_pct:.4f}s")
    print(f"   Prompts per second: {prompts_per_second:.2f}")
    print(f"   Prompts in queue: {q.qsize()}")
    print(f"   Server/Unprocessed Prompts: {unprocessed_server_prompts}")
    print(f"   Client/Requested prompts per second: {(len(metrics) - q.qsize()) / args.time:.2f}")
    print(f"   Client/Created prompts per second: {len(metrics) / args.time:.2f}")
    print(f"   Client/Total prompts: {len(metrics)}")

    writer.add_scalar("Throughput", throughput, batch_size)
    writer.add_scalar("TTFT/Median", ttft_median, batch_size)
    writer.add_scalar("TTFT/99th Percentile", ttft_99_pct, batch_size)
    writer.add_scalar("ITL/Median", itl_median, batch_size)
    writer.add_scalar("ITL/99th Percentile", itl_99_pct, batch_size)
    writer.add_scalar("Latency/Median", latency_median, batch_size)
    writer.add_scalar("Latency/99th Percentile", latency_99_pct, batch_size)
    writer.add_scalar("Decode Time/Median", decode_median, batch_size)
    writer.add_scalar("Decode Time/99th Percentile", decode_99_pct, batch_size)
    writer.add_scalar("Server/Unprocessed Prompts", unprocessed_server_prompts, batch_size)
    writer.add_scalar("Server/Processed Prompts per second", prompts_per_second, batch_size)
    writer.add_scalar("Client/Time in Queue Median", time_in_queue_median, batch_size)
    writer.add_scalar("Client/Time in Queue 99th Percentile", time_in_queue_99_pct, batch_size)
    writer.add_scalar("Client/Unprocessed Prompts", q.qsize(), batch_size)
    writer.add_scalar("Client/Requested prompts per second", (len(metrics) - q.qsize()) / args.time, batch_size)
    writer.add_scalar("Client/Created prompts per second", len(metrics) / args.time, batch_size)
    writer.add_scalar("Client/Total prompts", len(metrics), batch_size)

writer = SummaryWriter(f"results/tensorboard_logs/{'chunked_prefill' if args.using_chunked_prefill else 'wo_chunked_prefill'}/")
nt = 12
for batch_size in tqdm([2**k for k in range(12)]):
    asyncio.run(run_experiment(batch_size, writer, nt))
writer.close()