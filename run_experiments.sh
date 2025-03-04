#! /bin/bash
BASE_DIR=/mnt/llama.cpp/llama2-7B-hf/
tb_base_dir=./tensorboard_logs/

model=facebook/opt-125M

python3 cached_vs_uncached_experiment.py --enable_cache --model $model  --tokenizer_path ${BASE_DIR} --tensorboard_log_pathname ${tb_base_dir}cached/
python3 cached_vs_uncached_experiment.py --model $model --tokenizer_path ${BASE_DIR} --tensorboard_log_pathname ${tb_base_dir}uncached/