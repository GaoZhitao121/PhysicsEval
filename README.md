```bash
nohup env CUDA_VISIBLE_DEVICES=1 \
vllm serve /root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_0101/models/psp_round_1 \
  --port 8001 \
  --max-model-len 15360 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --served-model-name Qwen2.5_7B_KTO_staggered_0101_round_1 \
  > vllm_Qwen2.5_7B_KTO_staggered_0101_round_1.log 2>&1 &
  
python PROPOSER.py \
    --base_url http://localhost:8001/v1 \
    --model Qwen2.5_7B_KTO_staggered_0101_round_1 \
    --api_key token-vllm \
    --concurrency 20

## move BASE SOLUTION/SOLUTIONS/***.jsonl to EVALUATIONS

python eval_gpt.py ***.jsonl
```




