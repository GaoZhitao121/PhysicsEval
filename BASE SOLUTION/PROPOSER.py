import json
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ================= 配置与常量 =================
MAX_TIME_LIMIT = 180  # 每个请求的超时时间（秒）
file_lock = threading.Lock() # 用于多线程安全写入文件的锁

def parse_args():
    """处理命令行参数"""
    parser = argparse.ArgumentParser(description="使用 vLLM 或 OpenAI 兼容接口进行物理题目推理")
    parser.add_argument("--api_key", type=str, default="token-vllm", help="API 密钥")
    parser.add_argument("--base_url", type=str, required=True, help="接口地址，例如 http://localhost:8000/v1")
    parser.add_argument("--model", type=str, required=True, help="模型名称，需与 vLLM 启动时一致")
    parser.add_argument("--concurrency", type=str, default=10, help="并发请求数 (默认: 10)")
    parser.add_argument("--input", type=str, default="test set.json", help="输入题目文件路径")
    return parser.parse_args()

def sanitize_file_name(name: str):
    _forbidden_chars = "<>:\"/\\|?* "
    for _c in _forbidden_chars:
        name = name.replace(_c, "_")
    return name

def get_solution(client, model, problem):
    """请求模型获取答案"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (f"You are an expert on Physics. You solve problems step by step while maintaining logical consistency. "
                                f"Solve the following Physics problem: {problem} "
                                "Finally, write the final answers in brief. Make sure you write all equations in LaTeX.")
                }
            ],
            timeout=MAX_TIME_LIMIT
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"请求出错: {e}")
        return None

def process_single_problem(client, model, problem, output_file):
    """单个题目的处理逻辑"""
    ID = problem['Problem_ID']
    solution = get_solution(client, model, problem['problem'])
    
    if solution:
        data = {
            'Problem_ID': ID,
            'problem': problem['problem'],
            'ai_solution': solution,
            'elaborated_solution_steps': problem['elaborated_solution_steps']
        }
        
        # 线程安全写入
        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data) + '\n')
        return True
    return False

def main():
    args = parse_args()
    
    # 初始化客户端
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    os.makedirs("./SOLUTIONS", exist_ok=True)
    output_file = f"./SOLUTIONS/proposed_solution_by_{sanitize_file_name(args.model)}.jsonl"

    # 读取题目数据
    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        return
    with open(args.input, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # 检查已完成题目实现断点续传
    completed_problems = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    completed_problems.add(json.loads(line)['Problem_ID'])
                except: continue
    
    tasks = [p for p in problems if p['Problem_ID'] not in completed_problems]
    print(f"总计: {len(problems)} 题 | 已完成: {len(completed_problems)} 题 | 待处理: {len(tasks)} 题")
    print(f"并发数设置为: {args.concurrency}")

    error_count = 0
    # 使用线程池进行并发请求
    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as executor:
        future_to_id = {executor.submit(process_single_problem, client, args.model, p, output_file): p['Problem_ID'] for p in tasks}
        
        for i, future in enumerate(as_completed(future_to_id), start=1):
            prob_id = future_to_id[future]
            try:
                if not future.result():
                    error_count += 1
                    print(f"[{i}/{len(tasks)}] 失败: {prob_id}")
                else:
                    print(f"[{i}/{len(tasks)}] 成功: {prob_id}")
            except Exception as e:
                error_count += 1
                print(f"[{i}/{len(tasks)}] 任务异常: {prob_id} - {e}")

    if error_count:
        print(f"\n任务结束，共有 {error_count} 个错误。请重新运行以处理失败题目。")
    else:
        print("\n所有题目处理成功！")

if __name__ == "__main__":
    main()

# python PROPOSER.py \
#     --base_url http://localhost:8001/v1 \
#     --model Qwen2.5-7B-Instruct \
#     --api_key token-vllm \
#     --concurrency 20

