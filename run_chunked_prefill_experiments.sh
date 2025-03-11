model=llama/Llama-2-7B-hf-F16.gguf
time=1
sleep_time=0.1

# without enable chunked prefill
vllm serve ${model} --port 8000 &
PID1=$!
echo $PID1
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time ${sleep_time}

kill $PID1

# with enable chunked prefill
vllm serve $model --port 8000 --enable-chunked-prefill &
PID2=$!
echo $PID2
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time ${sleep_time} --using_chunked_prefill

kill $PID2
