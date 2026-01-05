#!/bin/bash

# ================= 配置区域 =================
MODELS=(
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_a_0104/models/psp_round_5"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_c_0104/models/psp_round_5"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_1"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_2"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_3"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_4"
    "/root/codespace/gaozhitao/PSP/experiments/Qwen2.5_7B_KTO_staggered_wo_q_0104/models/psp_round_5"
)

GPU_ID=3
PORT=8003
MAX_MODEL_LEN=15360
GPU_UTIL=0.9
CONCURRENCY=20
# ============================================

for MODEL_PATH in "${MODELS[@]}"; do
    # 1. 自动解析名称 (提取实验名和轮次名)
    ROUND_NAME=$(basename "$MODEL_PATH")
    EXP_NAME=$(basename "$(dirname "$(dirname "$MODEL_PATH")")")
    # 格式化名称: EXP_NAME_ROUND_NAME
    MODEL_NAME="${EXP_NAME}_${ROUND_NAME}"

    echo "#########################################################"
    echo "开始执行 PROPOSER 测评: $MODEL_NAME"
    echo "使用端口: $PORT | 使用 GPU: $GPU_ID"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "#########################################################"

    # 2. 预清理：确保端口 8003 没有被残留进程占用
    pkill -f "vllm.*--port $PORT" || true
    sleep 2

    # 3. 启动 vLLM Serve (后台运行)
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$MODEL_PATH" \
      --port $PORT \
      --max-model-len $MAX_MODEL_LEN \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization $GPU_UTIL \
      --served-model-name "$MODEL_NAME" \
      > "vllm_${MODEL_NAME}.log" 2>&1 &
    
    VLLM_PID=$!

    # 4. 等待 vLLM 服务就绪 (健康检查)
    echo "正在加载权重，等待服务启动..."
    while ! curl -s http://127.0.0.1:$PORT/v1/models > /dev/null; do
        if ! ps -p $VLLM_PID > /dev/null; then
            echo "错误: vLLM 进程意外退出，请检查日志 vllm_${MODEL_NAME}.log"
            continue 2 # 跳过当前模型
        fi
        sleep 10
    done
    echo "服务已就绪，开始运行 PROPOSER.py..."

    # 5. 运行 PROPOSER.py (前台运行，直到结束)
    python PROPOSER.py \
        --base_url "http://localhost:$PORT/v1" \
        --model "$MODEL_NAME" \
        --api_key "token-vllm" \
        --concurrency $CONCURRENCY \
        > "proposer_${MODEL_NAME}.log" 2>&1

    # 6. 清理：释放 GPU 3 的资源
    echo "测评结束，正在杀掉 vLLM 进程并回收显存..."
    pkill -f "vllm.*--port $PORT" || true
    
    # 强制等待显存回收到系统 (对于 7B 模型 15s 足够，对于 32B 建议增加)
    sleep 20
    
    echo "模型 $MODEL_NAME 测评流程已完成。"
    echo "---------------------------------------------------------"
done

echo "所有模型的 PROPOSER 测评任务全部执行完毕！"

# nohup bash auto_proposer_eval.sh > proposer_summary.log 2>&1 &