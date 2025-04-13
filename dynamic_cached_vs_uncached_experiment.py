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
import torch
import matplotlib.pyplot as plt



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



def configure_launcher(args,enable_apc,final_path = ""):
    root_dir = "results/"


    llm = LLM(
        model=args.model,
        task="generate",
        tokenizer=args.tokenizer_path,
        enable_prefix_caching=enable_apc,
    )

    writer = SummaryWriter(f"{root_dir}tensorboard_logs/{'cached' if enable_apc else 'uncached' }/{final_path}")
    stop_event = threading.Event()
    thread = threading.Thread(target=monitor_gpu_memory, args=(writer, stop_event))
    thread.start()
    return llm, writer, thread, stop_event
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", help="")
    parser.add_argument("--tokenizer_path", help="")
    parser.add_argument("--reverse",action="store_true", help="")

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    ds = load_dataset("fka/awesome-chatgpt-prompts")
    prompts = ds["train"]['prompt']

    all_prompt_idx = []
    random.seed(1234)

    batch_size = 1
    while batch_size <= 512:
        idxs = [random.randint(0, len(prompts) - 1) for _ in range(batch_size)]
        all_prompt_idx.append(idxs)
        batch_size *= 2

    pct_reused_prompts_list = dict([[len(prompt_idxs), calc_pct_reused_prompts(prompt_idxs)] for i,prompt_idxs in enumerate(all_prompt_idx)])

    print("Batch size -> Pct reused prompts:")
    print(pct_reused_prompts_list)
    print()


    for enable_apc in [False, True]:
        llm, writer, thread ,event = configure_launcher(args, enable_apc,"dynamic_batch/")
        #warmup
        outputs = llm.generate(prompts[:256], SamplingParams(temperature=0.8, top_p=0.95))
        torch.cuda.empty_cache()
        llm.reset_prefix_cache()

        writer.add_text("pct_reused_prompts", str(pct_reused_prompts_list))
        for prompt_idx in all_prompt_idx:
            batch_size = len(prompt_idx)
            cur_prompts = [prompts[i] for i in prompt_idx]

            init_time = time.time()
            outputs = llm.generate(cur_prompts, SamplingParams(temperature=0.8, top_p=0.95))
            final_time = time.time()

            num_decoded_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
            num_tokens_prompt = sum(len(tokenizer.tokenize(prompt)) for prompt in cur_prompts)

            elapsed_time = final_time - init_time
            throughput = (num_tokens_prompt + num_decoded_tokens) / elapsed_time

            writer.add_scalar("Latency(s) x Batch size", elapsed_time, batch_size)
            writer.add_scalar("Throughput(tok/s) x Batch size", throughput, batch_size)
            torch.cuda.empty_cache()
            llm.reset_prefix_cache()


        event.set()
        thread.join()
        writer.close()
        del llm  

