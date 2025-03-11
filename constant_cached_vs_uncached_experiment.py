import time
from dynamic_cached_vs_uncached_experiment import configure_launcher, get_args
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch

if __name__ == "__main__":
    args = get_args()
    tenk_ds = load_dataset("data-is-better-together/10k_prompts_ranked")

    for enable_apc in [False, True]:
        for batch_size in [2**i for i in range(4,7)]:
            cur_prompts = tenk_ds["train"]['prompt'][:batch_size]
            llm, writer, thread ,event = configure_launcher(args, enable_apc,f"constant_batch_{batch_size}/")
            for i in range(batch_size):
                cur_prompts[i] = cur_prompts[0]
                init_time = time.time()
                outputs = llm.generate(cur_prompts, SamplingParams(temperature=0.8, top_p=0.95))
                final_time = time.time()

                num_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
                elapsed_time = final_time - init_time
                throughput = num_tokens / elapsed_time

                writer.add_scalar("latency(s)_vs_batch_size", elapsed_time, i)
                writer.add_scalar("throughput(tok/s)_vs_batch_size", throughput, i)

            event.set()
            thread.join()
            writer.close()
            del llm  
            torch.cuda.empty_cache()