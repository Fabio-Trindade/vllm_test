import GPUtil
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
from utils import create_dir

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tensorboard_log_pathname", help="")
args = parser.parse_args()

tensorboard_log_path = args.tensorboard_log_pathname

create_dir(tensorboard_log_path)
writer = SummaryWriter(tensorboard_log_path)

def monitor_gpu_memory(interval = 1):
    step = 0 
    try:
        while True:
            gpus = GPUtil.getGPUs()

            for gpu in gpus:
                memory_used = gpu.memoryUsed    
                memory_util = gpu.memoryUtil    
            
                writer.add_scalar(f"GPU_{gpu.id}/memory_used", memory_used, step)
                writer.add_scalar(f"GPU_{gpu.id}/memory_utilization", memory_util * 100, step)
            
            step += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Interrupted.")


monitor_gpu_memory()

writer.close()