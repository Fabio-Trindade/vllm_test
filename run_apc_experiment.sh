#! /bin/bash
BASE_DIR=/home/fabio.ramos/vllm_test
tb_base_dir=./results/tensorboard_logs/

model=./llama/Llama-2-7B-hf-F16.gguf

python3 dynamic_cached_vs_uncached_experiment.py  --model $model  --tokenizer_path ${BASE_DIR}/llama --tensorboard_log_pathname ${tb_base_dir}cached/

python3 constant_cached_vs_uncached_experiment.py --model $model  --tokenizer_path ${BASE_DIR}/llama --tensorboard_log_pathname ${tb_base_dir}cached/
python3 constant_cached_vs_uncached_experiment.py --model $model --tokenizer_path ${BASE_DIR}/llama --tensorboard_log_pathname ${tb_base_dir}uncached/ --reverse
