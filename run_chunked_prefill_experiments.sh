model=llama/Llama-2-7B-hf-F16.gguf
time=10
sleep_time_queue=0.0
sleep_time_request=0.5
model_seq_len=4096
# without enable chunked prefill
vllm serve ${model} --port 8000 &
PID1=$!
echo $PID1
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time-queue ${sleep_time_queue} --sleep-time-request ${sleep_time_request} --model-seq-len ${model_seq_len}

kill $PID1

# with enable chunked prefill
vllm serve $model --port 8000 --enable-chunked-prefill &
PID2=$!
echo $PID2
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time-queue ${sleep_time_queue} --sleep-time-request ${sleep_time_request} --model-seq-len ${model_seq_len} --using_chunked_prefill

kill $PID2
