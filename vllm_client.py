import argparse
import asyncio
import httpx
import time
import random
from datasets import load_dataset

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default="facebook/opt-125M", help="")
parser.add_argument("--num_threads", type=int, default=5, help="")
parser.add_argument("--time", type=int, default=60, help="")
parser.add_argument("--sleep_time", type=float, default=0.1, help="")
args = parser.parse_args()

ds = load_dataset("fka/awesome-chatgpt-prompts")
prompts = ds["train"]['prompt']
vllm_server_url = "http://localhost:8000/v1/completions"

total_requests = 0
total_processed_tokens = 0
lock = asyncio.Lock()

async def send_data(prompts, client, max_tokens):
    global total_requests, total_processed_tokens

    prompt = random.choice(prompts)
    request = {
        "model": args.model,
        "prompt": prompt,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False
    }

    try:
        response = await client.post(vllm_server_url, json=request)
        response_json = response.json()
        completion_tokens = response_json.get("usage", {}).get("completion_tokens", 0)

        async with lock:
            total_requests += 1
            total_processed_tokens += completion_tokens

        print(f"Response: {response.status_code}, Tokens: {completion_tokens}")

    except httpx.RequestError as e:
        print(f"Request error: {e}")

async def main():
    async with httpx.AsyncClient() as client:
        init_time = time.time()
        elapsed_time = 0
        tasks = []

        while elapsed_time < args.time:
            for _ in range(args.num_threads):
                tasks.append(asyncio.create_task(send_data(prompts, client, max_tokens=512)))
            
            elapsed_time = time.time() - init_time
            await asyncio.sleep(args.sleep_time)  

        await asyncio.gather(*tasks)  

init_time = time.time()
asyncio.run(main())
total_time = time.time() - init_time

throughput = total_processed_tokens / total_time
inter_token_latency = 1 / throughput

print("Processed tokens:", total_processed_tokens)
print("Total time:", total_time)
print("Throughput (tokens/sec):", throughput)
print("Inter-token latency (sec/token):", inter_token_latency)
print("Total requests:", total_requests)
