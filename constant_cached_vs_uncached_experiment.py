import time
from dynamic_cached_vs_uncached_experiment import configure_launcher, get_args
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer  

if __name__ == "__main__":
    args = get_args()
    tenk_ds = load_dataset("data-is-better-together/10k_prompts_ranked")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    for enable_apc in [False, True]:
        for batch_size in [2**i for i in range(4,12)]:
            cur_prompts = tenk_ds["train"]['prompt'][:batch_size]
            llm, writer, thread ,event = configure_launcher(args, enable_apc,f"constant_batch_{batch_size}/")
            for i in range(batch_size):
                cur_prompts[i] = cur_prompts[0]
                init_time = time.time()
                outputs = llm.generate(cur_prompts, SamplingParams(temperature=0.8, top_p=0.95))
                final_time = time.time()

                    
                num_decoded_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
                num_tokens_prompt = sum(len(tokenizer.tokenize(prompt)) for prompt in cur_prompts)
                elapsed_time = final_time - init_time
                throughput = (num_tokens_prompt + num_decoded_tokens) / elapsed_time

                writer.add_scalar("Latency(s) x Batch size", elapsed_time, batch_size)
                writer.add_scalar("Throughput(tok/s) x Batch size", throughput, batch_size)
            event.set()
            thread.join()
            writer.close()
            del llm  
            torch.cuda.empty_cache()