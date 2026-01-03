import os
import json
import requests
import logging
import sys
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区域 =================
os.environ["OPENAI_BASE_URL"] = "http://yy.dbh.baidu-int.com/v1"
os.environ["OPENAI_API_KEY"] = "sk-s8NRXPyriHxlW5UCXWTKAaK6dYMzHrw5GzHxQn3y9zPahm2n"

OPENAI_API_URL = f"{os.environ['OPENAI_BASE_URL'].rstrip('/')}/chat/completions"
MODEL_NAME = "gpt-4.1" 
API_KEY = os.environ["OPENAI_API_KEY"]

MAX_WORKERS = 20  # 并发数
API_TIMEOUT = 180

# 论文定义的权重 
PPS_WEIGHTS = {
    "mathematical_accuracy": 0.30,
    "logical_consistency": 0.25,
    "formulas_principles": 0.20,
    "completeness": 0.10,
    "assumptions_made": 0.10,
    "clarity_and_coherence": 0.05
}
# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_evaluation_prompt(problem_id, elaborated_solution, ai_solution):
    """补全后的完整打分 Prompt [cite: 446-510]"""
    return f"""You are an expert physics problem evaluator. Your task is to meticulously and STRICTLY evaluate an AI-generated solution against a ground-truth solution.

Evaluation Categories and Scoring Guidelines (Score 1-5):
1. mathematical_accuracy (Weight 0.30): Calculation correctness, units, and numerical values.
2. logical_consistency (Weight 0.25): Step-by-step reasoning and physics logic.
3. completeness (Weight 0.10): Addressing all parts of the problem.
4. clarity_and_coherence (Weight 0.05): Organization and use of terminology.
5. formulas_principles (Weight 0.20): Correct identification and application of physical laws.
6. assumptions_made (Weight 0.10): Explicit and justified assumptions.
7. overall_correctness (Score 0-10): Overall fidelity to ground truth.

Problem ID: {problem_id}
Ground Truth Elaborated Steps:
{elaborated_solution}

AI-Generated Solution:
{ai_solution}

Provide evaluation STRICTLY as a JSON object.
{{
    "problem_id": "{problem_id}",
    "mathematical_accuracy": <1-5>,
    "logical_consistency": <1-5>,
    "completeness": <1-5>,
    "clarity_and_coherence": <1-5>,
    "formulas_principles": <1-5>,
    "assumptions_made": <1-5>,
    "overall_correctness": <0-10>
}}"""

def calculate_pps(eval_dict):
    """根据论文公式计算归一化的 PPS 指标 (0-100) [cite: 181, 193]"""
    try:
        raw_weighted_sum = 0
        for metric, weight in PPS_WEIGHTS.items():
            # 默认得分为 1（最低分）以防字段缺失
            score = float(eval_dict.get(metric, 1))
            raw_weighted_sum += score * weight
        
        # 归一化公式: ((raw - min) / (max - min)) * 100
        # min = 1.0, max = 5.0
        normalized_pps = ((raw_weighted_sum - 1.0) / 4.0) * 100
        return round(normalized_pps, 2)
    except Exception as e:
        logger.error(f"PPS 计算出错: {e}")
        return 0.0

def call_openai_api(prompt: str):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], 200
    except Exception as e:
        logger.error(f"API Error: {e}")
        return None, 500

def evaluate_item(item):
    problem_id = item.get('Problem_ID')
    prompt = create_evaluation_prompt(problem_id, item.get('elaborated_solution_steps', ''), item.get('ai_solution', ''))
    res_text, _ = call_openai_api(prompt)
    if res_text:
        try:
            match = re.search(r"(\{[\s\S]*\})", res_text)
            eval_json = json.loads(match.group(1)) if match else json.loads(res_text)
            
            # --- 加入 PPS 指标计算 ---
            pps_val = calculate_pps(eval_json)
            eval_json['pps_score'] = pps_val
            
            item['gpt4_evaluation'] = eval_json
            return item
        except Exception as e:
            logger.error(f"解析结果失败 {problem_id}: {e}")
            return None
    return None

def process_file_concurrently(input_path: Path):
    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return

    output_path = input_path.parent / f"evaluated_{input_path.stem}.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        items = [json.loads(line) for line in f if line.strip()]

    logger.info(f"开始评测: {input_path.name} ({len(items)}题)")
    results = []
    total_pps = 0
    count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(evaluate_item, item): item for item in items}
        for i, future in enumerate(as_completed(future_to_item), 1):
            res = future.result()
            if res:
                results.append(res)
                total_pps += res['gpt4_evaluation']['pps_score']
                count += 1
            if i % 10 == 0: logger.info(f"进度: {i}/{len(items)}")

    # 保存包含 pps_score 的结果文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if count > 0:
        avg_pps = total_pps / count
        logger.info(f"完成！该文件平均 PPS 得分为: {avg_pps:.2f} / 100")
        logger.info(f"详细结果保存至: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_file_concurrently(Path(sys.argv[1]))
    else:
        for f in Path('.').glob('*.jsonl'):
            process_file_concurrently(f)

# python eval_gpt.py eval_input.jsonl

