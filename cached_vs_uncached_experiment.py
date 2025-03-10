import argparse
import random
import threading
from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import io
from utils import create_dir, calc_dataset_stats, hash_string_sha256
from collections import Counter
import GPUtil
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", help="")
parser.add_argument("--tokenizer_path", help="")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

def monitor_gpu_memory(writer, stop_event):
    step = 0 
    while not stop_event.is_set():
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            memory_used = gpu.memoryUsed    
            memory_util = gpu.memoryUtil    

            writer.add_scalar(f"GPU_{gpu.id}/memory_used", memory_used, step)
            writer.add_scalar(f"GPU_{gpu.id}/memory_utilization", memory_util * 100, step)
        
        step += 1
        time.sleep(0.1)

def calc_pct_reused_prompts(prompts_idxs):
    unique_values = Counter(prompts_idxs)
    total_prompts_size = len(prompts_idxs)
    return (total_prompts_size - len(unique_values)) / total_prompts_size

def calc_pct_reused_blocks(tokenizer, prompts, block_size):
    hashes = set()
    total_blocks = 0
    for prompt in prompts:
        cur_string = ""
        real_prompt = tokenizer(prompt)["input_ids"]
        prompt_len = len(real_prompt)
        for start_idx in range(0, prompt_len, block_size):
            total_blocks += 1
            final_idx = min(start_idx + block_size, prompt_len)
            cur_string += ' '.join(map(str, real_prompt[start_idx:final_idx])) + ' '
            cur_hash = hash_string_sha256(cur_string)
            hashes.add(cur_hash)
    return (total_blocks - len(hashes)) / total_blocks

ds = load_dataset("fka/awesome-chatgpt-prompts")
prompts = ds["train"]['prompt']

all_prompt_idx = []
batch_size = 1
random.seed(1234)
while batch_size <= 512:
    idxs = [random.randint(0, len(prompts) - 1) for _ in range(batch_size)]
    all_prompt_idx.append(idxs)
    batch_size *= 2

pct_reused_prompts_list = dict([[len(prompt_idxs),calc_pct_reused_prompts(prompt_idxs)] for i,prompt_idxs in enumerate(all_prompt_idx)])

print("Batch size -> Pct reused prompts:")
print(pct_reused_prompts_list)
print()

root_dir = "results/"

dir_list = ["cached", "uncached"]

def configure_launcher(dir,block_size):
    enable_apc = (dir == "cached")

    llm = LLM(
        model=args.model,
        task="generate",
        tokenizer=args.tokenizer_path,
        enable_prefix_caching=enable_apc,
        block_size=block_size if enable_apc else None
    )

    writer = SummaryWriter(f"{root_dir}tensorboard_logs/{dir}/block_size_{block_size if enable_apc else '0'}/")
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_gpu_memory, args=(writer, stop_event))
    thread.start()
    return llm, writer, thread, enable_apc, stop_event

for block_size in [2**i for i in range(5,9)]:
    for dir in dir_list:
        llm, writer, thread, enable_apc,event = configure_launcher(dir,block_size)
        for prompt_idx in all_prompt_idx:
            batch_size = len(prompt_idx)
            cur_prompts = [prompts[i] for i in prompt_idx]

            pct_reused_prompts = pct_reused_prompts_list[len(prompt_idx)]

            init_time = time.time()
            outputs = llm.generate(cur_prompts, SamplingParams(temperature=0.8, top_p=0.95))
            final_time = time.time()

            num_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            elapsed_time = final_time - init_time
            throughput = num_tokens / elapsed_time

            writer.add_scalar("latency(s)_vs_batch_size", elapsed_time, batch_size)
            writer.add_scalar("throughput(tok/s)_vs_batch_size", throughput, batch_size)

            if enable_apc:
                pct_reused_blocks = calc_pct_reused_blocks(tokenizer, cur_prompts, block_size)
                writer.add_scalar("latency(s)_vs_reused_blocks(%)", elapsed_time, pct_reused_blocks)
                writer.add_scalar("throughput(tok/s)_vs_reused_blocks(%)", throughput, pct_reused_blocks)

        event.set()
        thread.join()
        writer.close()

        if not enable_apc:
            break
    if not enable_apc:
        break

for dir in dir_list:
    batch_size = 1
    llm, writer, thread, enable_apc,event = configure_launcher(dir,115)
    count = 0 
    while batch_size <= 1501:
        idxs = [random.randint(0, batch_size - 1) for _ in range(batch_size)]
        pct_reused_prompts = calc_pct_reused_prompts(idxs)
        all_prompt_idx.append(idxs)
        batch_size += 50
        writer.add_scalar("latency(s)_vs_reused_prompts(%)", elapsed_time, batch_size)
        writer.add_scalar("throughput(tok/s)_vs_reused_prompts(%)", throughput, batch_size)

    event.set()
    thread.join()
    writer.close()

