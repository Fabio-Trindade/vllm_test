import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset
import os
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import io
from utils import create_dir
parser = argparse.ArgumentParser(description="")

parser.add_argument("--enable_cache", action="store_true", help="")
parser.add_argument("--model", help="")
parser.add_argument("--tokenizer_path", help="")
parser.add_argument("--tensorboard_log_pathname", help="")
args = parser.parse_args()


tensorboard_log_path = args.tensorboard_log_pathname

create_dir(tensorboard_log_path)

ds = load_dataset("fka/awesome-chatgpt-prompts")

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = ds["train"]['prompt']
batch_size = 1
model_path = args.model
tokenizer_path = args.tokenizer_path

llm = LLM(
    model=model_path,
    task="generate",
    tokenizer=tokenizer_path,
    enable_prefix_caching=args.enable_cache,  
)

lambda: llm.apply_model(lambda model: print(model.__class__))
writer = SummaryWriter(tensorboard_log_path)

while batch_size < len(prompts):
    cur_prompts = prompts[:batch_size]

    init_time = time.time()
    outputs = llm.generate(cur_prompts, sampling_params)
    final_time = time.time()

    num_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    elapsed_time = final_time - init_time
    throughput = num_tokens / elapsed_time

    writer.add_scalar("time_vs_batch_size", final_time - init_time, batch_size)
    writer.add_scalar("throughput_vs_batch_size", throughput, batch_size)
    batch_size *= 2

writer.close()
