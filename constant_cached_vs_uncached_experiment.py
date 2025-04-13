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
        for batch_size in [2**i for i in range(4,10)]:
            if not args.reverse:
                cur_prompts = tenk_ds["train"]['prompt'][:batch_size]
            else:
                cur_prompts = [tenk_ds["train"]["prompt"][0] for _ in range(batch_size)]
            llm, writer, thread ,event = configure_launcher(args, enable_apc,f"constant_batch_{batch_size}/{"reversed" if args.reverse else "not_reversed"}/")
            outputs = llm.generate(tenk_ds["train"]["prompt"][:256], SamplingParams(temperature=0.8, top_p=0.95))
            llm.reset_prefix_cache()
            torch.cuda.empty_cache()

            for i in range(batch_size):
                if not args.reverse:
                    cur_prompts[i] = cur_prompts[0]
                else:
                    cur_prompts[i] = tenk_ds["train"]['prompt'][i]

                init_time = time.time()
                outputs = llm.generate(cur_prompts, SamplingParams(temperature=0.8, top_p=0.95))
                final_time = time.time()

                    
                num_decoded_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
                num_tokens_prompt = sum(len(tokenizer.tokenize(prompt)) for prompt in cur_prompts)
                elapsed_time = final_time - init_time
                throughput = (num_tokens_prompt + num_decoded_tokens) / elapsed_time

                writer.add_scalar("Latency(s) x Batch size", elapsed_time, i)
                writer.add_scalar("Throughput(tok/s) x Batch size", throughput, i)
                llm.reset_prefix_cache()
                torch.cuda.empty_cache()
            event.set()
            thread.join()
            writer.close()
            llm = None
