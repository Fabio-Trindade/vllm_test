model=facebook/opt-125M
time=1
sleep_time=0.1

# without enable chunked prefill
vllm serve ${model} --port 8000 &
PID1=$!
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time ${sleep_time}

# Mata o servidor após o uso
kill $PID1

# with enable chunked prefill
vllm serve $model --port 8000 --enable-chunked-prefill &
PID2=$!
sleep 120
python3 vllm_client.py  --model ${model} --time ${time} --sleep-time ${sleep_time} --using_chunked_prefill

kill $PID2
