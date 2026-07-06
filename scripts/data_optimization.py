#!/usr/bin/env python3
"""
Search-R1 数据端优化模块
========================

在检索召回优化之外，从数据端提供以下增强：

  1. 查询扩展 (Query Expansion) - 提升查询多样性
  2. 数据难度分层 (Difficulty Stratification) - 支持课程学习
  3. 检索质量预评估 (Retrieval Quality Pre-eval) - 筛选高质量数据
  4. 查询意图分类 (Query Intent Classification) - 匹配检索策略
  5. 数据增强 (Contrastive Augmentation) - 对比学习数据构造
  6. 答案一致性验证 (Answer Consistency Check) - 数据去噪

设计原则：
  - 所有优化为新增模块，不修改原有数据处理流程
  - 支持选择性地启用/禁用各项优化
  - 输出格式与原数据格式兼容

用法:
  python scripts/data_optimization.py --input_path ./data/nq_search/train.parquet \
                                       --output_path ./data/nq_search/train_optimized.parquet \
                                       --enable_expansion \
                                       --enable_difficulty \
                                       --enable_filtering

依赖: datasets, numpy
"""

import json
import os
import re
import argparse
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


# ============================================================
# 配置
# ============================================================

@dataclass
class DataOptimizationConfig:
    """数据优化配置"""
    # 查询扩展
    enable_expansion: bool = False
    expansion_methods: List[str] = field(default_factory=lambda: ["synonym", "back_translate_like", "question_rephrase"])
    expansion_factor: int = 2  # 每条数据扩展倍数

    # 难度分层
    enable_difficulty: bool = False
    difficulty_levels: int = 3  # easy / medium / hard
    difficulty_method: str = "heuristic"  # heuristic / embedding

    # 数据过滤
    enable_filtering: bool = False
    min_question_length: int = 10
    max_question_length: int = 500
    min_answer_length: int = 1

    # 查询意图分类
    enable_intent: bool = False

    # 对比增强
    enable_contrastive: bool = False
    contrastive_negative_count: int = 2

    # 输出
    output_format: str = "parquet"  # parquet / jsonl


# ============================================================
# 1. 查询扩展 (Query Expansion)
# ============================================================

SYNONYM_DICT = {
    # 常用词同义词
    "what is": ["define", "explain what", "tell me about", "what does", "describe"],
    "how to": ["in what way can", "what is the method to", "explain how to", "steps to"],
    "when did": ["at what time did", "in which year did", "what was the date of", "during when did"],
    "where is": ["what is the location of", "in which place is", "where can I find", "locate"],
    "who is": ["tell me about", "who was", "give information about", "identify"],
    "why does": ["for what reason does", "what causes", "explain why", "what is the reason"],
    "which": ["what", "select the", "pick the", "choose the"],
    "name the": ["list the", "identify the", "what are the", "tell me the"],
    "difference between": ["distinction between", "compare", "contrast between", "how do X and Y differ"],
    "similar to": ["analogous to", "comparable to", "like", "related to"],
    # 领域特定同义词
    "capital": ["capital city", "seat of government", "administrative center"],
    "population": ["number of people", "inhabitants", "residents", "demographics"],
    "founded": ["established", "created", "started", "originated"],
    "discovered": ["found", "identified", "uncovered", "detected"],
}


def synonym_expansion(question: str, n_variants: int = 2) -> List[str]:
    """基于同义词词典的查询扩展"""
    variants = []
    q_lower = question.lower()

    for original, syns in SYNONYM_DICT.items():
        if original in q_lower:
            for syn in syns[:n_variants]:
                variant = question
                # 替换第一个匹配（保持大小写）
                idx = q_lower.find(original)
                if idx >= 0:
                    variant = question[:idx] + syn + question[idx + len(original):]
                if variant != question:
                    variants.append(variant)
            break  # 只应用第一个匹配的转换

    return list(set(variants))[:n_variants]


def back_translate_like_expansion(question: str, n_variants: int = 2) -> List[str]:
    """
    模拟回译增强：通过句式变换模拟回译效果

    注意：实际回译需要翻译模型，这里使用规则模拟
    """
    variants = []

    # 规则1: 被动语态 ↔ 主动语态
    if " was " in question.lower():
        # was discovered by X → X discovered
        m = re.search(r"(.*?) was (\w+) by (.*?)(\?|$)", question, re.IGNORECASE)
        if m:
            variants.append(f"{m.group(3).strip()} {m.group(2)} {m.group(1).strip()}?")
    elif " by " in question.lower() and "?" in question:
        m = re.search(r"(.*?) by (.*?)\?", question, re.IGNORECASE)
        if m:
            variants.append(f"{m.group(2).strip()} {m.group(1).strip()}?")

    # 规则2: What is X that Y? → Y is what?
    m = re.search(r"[Ww]hat (?:is|are|was|were) (.*?) (?:that|which) (.*?)\?", question)
    if m:
        variants.append(f"{m.group(2).strip()} - what is it called?")

    # 规则3: When/Where/Who → 重新表述
    rephrase_map = {
        "when": ["At what time", "In what year", "During which period"],
        "where": ["In what location", "At which place", "In which country"],
        "who": ["Which person", "What individual", "Whom"],
        "why": ["For what reason", "What is the cause", "What led to"],
    }
    for trigger, rephrases in rephrase_map.items():
        if question.lower().startswith(trigger):
            for rp in rephrases[:n_variants]:
                variants.append(rp + question[len(trigger):])
            break

    return list(set(variants))[:n_variants]


def question_rephrase(question: str, n_variants: int = 2) -> List[str]:
    """基于模板的查询改写"""
    variants = []

    # 模板1: 去掉/添加 "please", "tell me" 等
    if question.lower().startswith("tell me"):
        variants.append(question[8:].strip())
    if not question.lower().startswith("what") and not question.lower().startswith("which"):
        variants.append(f"Tell me: {question}")

    # 模板2: 陈述句 ↔ 疑问句
    if "?" not in question:
        variants.append(f"{question}?")

    # 模板3: 简化/扩展
    words = question.split()
    if len(words) > 8:
        # 简化：取前8个关键词
        simplified = " ".join(words[:8]) + "?"
        variants.append(simplified)

    return list(set(variants))[:n_variants]


def expand_queries(questions: List[str], config: DataOptimizationConfig) -> List[List[str]]:
    """
    查询扩展主函数

    Returns:
        List[List[str]]: 每条原始查询的扩展变体列表
    """
    all_expanded = []
    for q in questions:
        variants = []
        for method in config.expansion_methods:
            if method == "synonym":
                variants.extend(synonym_expansion(q, config.expansion_factor))
            elif method == "back_translate_like":
                variants.extend(back_translate_like_expansion(q, config.expansion_factor))
            elif method == "question_rephrase":
                variants.extend(question_rephrase(q, config.expansion_factor))
        all_expanded.append(variants[:config.expansion_factor])
    return all_expanded


# ============================================================
# 2. 数据难度分层 (Difficulty Stratification)
# ============================================================

def compute_question_complexity(question: str) -> float:
    """基于启发式规则计算问题的复杂度 (0-1)"""
    score = 0.0

    # 1. 问题长度（信息量）
    words = question.split()
    if len(words) <= 5:
        score += 0.1
    elif len(words) <= 10:
        score += 0.3
    elif len(words) <= 20:
        score += 0.5
    else:
        score += 0.7

    # 2. 是否涉及多跳推理（包含多个实体或关系词）
    multi_hop_indicators = ["and", "or", "both", "between", "compare",
                             "difference", "relation", "after", "before",
                             "first", "then", "also", "another"]
    hop_count = sum(1 for w in multi_hop_indicators if w in question.lower().split())
    score += min(hop_count * 0.15, 0.3)

    # 3. 是否包含时间限定
    time_patterns = [r"\b\d{4}\b", r"\bin\s+\d{4}\b", r"\bsince\b", r"\bduring\b"]
    has_time = any(re.search(p, question, re.IGNORECASE) for p in time_patterns)
    if has_time:
        score += 0.1

    # 4. 是否包含否定/排除
    negation_indicators = ["not", "never", "except", "exclude", "without",
                           "other than", "unlike", "instead"]
    has_negation = any(w in question.lower() for w in negation_indicators)
    if has_negation:
        score += 0.15

    # 5. 专有名词数量（粗略估计：大写单词）
    proper_nouns = len([w for w in words if w[0].isupper()]) if words else 0
    score += min(proper_nouns * 0.05, 0.15)

    return min(score, 1.0)


def estimate_retrieval_difficulty(question: str, golden_answers: List[str]) -> Dict:
    """
    预估检索难度

    基于问题特征估计该问题是否需要依赖检索（vs 模型己知）

    返回:
        Dict with:
          - complexity: 问题复杂度分数
          - retrieval_dependency: 检索依赖度 (0=模型己知, 1=必须检索)
          - difficulty_level: easy/medium/hard
    """
    complexity = compute_question_complexity(question)

    # 检索依赖度估算
    # 规则: 复杂度高 + 答案长/涉及具体实体 → 更需要检索
    retrieval_dependency = complexity

    # 答案特征
    if golden_answers:
        avg_answer_len = np.mean([len(str(a)) for a in golden_answers])
        if avg_answer_len > 20:  # 长答案通常需要更多上下文
            retrieval_dependency = min(retrieval_dependency + 0.15, 1.0)
        if avg_answer_len <= 5:  # 短答案可能是知名实体
            retrieval_dependency = max(retrieval_dependency - 0.1, 0.0)

    # 判定难度等级
    if complexity < 0.35:
        level = "easy"
    elif complexity < 0.65:
        level = "medium"
    else:
        level = "hard"

    return {
        "complexity": round(complexity, 3),
        "retrieval_dependency": round(retrieval_dependency, 3),
        "difficulty_level": level,
    }


# ============================================================
# 3. 查询意图分类 (Query Intent Classification)
# ============================================================

QUERY_INTENT_PATTERNS = {
    "factoid_lookup": [
        r"^(what|who|when|where) (is|was|are|were|did|does)",
        r"^name the",
        r"^list the",
    ],
    "definition": [
        r"^(what|define|explain) (is|does|are) .*(mean|definition|referred)",
        r"^(what|define|explain) .* (term|concept|notion)",
    ],
    "comparison": [
        r"(?:difference|compare|contrast|similar|versus|vs\.?|better|worse)",
    ],
    "multi_hop_reasoning": [
        r"\band\b.*\band\b",  # 多个 and 可能表示多跳
        r"(?:first|then|after|before|subsequently)",
        r"(?:both|also|additionally|moreover|furthermore)",
    ],
    "temporal": [
        r"\b\d{4}\b",
        r"\b(in|during|since|before|after) (the )?\d{1,2}(st|nd|rd|th)? (century|decade|year)",
    ],
    "entity_centric": [
        r"(?:who|whom|whose)",
        r"(?:president|king|queen|ceo|founder|author|inventor|discoverer)",
    ],
}

def classify_query_intent(question: str) -> Dict:
    """基于模式匹配的查询意图分类"""
    intents = []
    scores = {}

    for intent, patterns in QUERY_INTENT_PATTERNS.items():
        match_count = sum(1 for p in patterns if re.search(p, question, re.IGNORECASE))
        if match_count > 0:
            intents.append(intent)
            scores[intent] = min(match_count / len(patterns), 1.0)

    # 默认意图
    if not intents:
        intents = ["factoid_lookup"]
        scores["factoid_lookup"] = 0.5

    primary_intent = max(intents, key=lambda x: scores.get(x, 0))

    recommended_retrieval = {
        "factoid_lookup": "bm25",       # BM25 对精确查找更好
        "definition": "dense",           # Dense 对语义定义更好
        "comparison": "hybrid",          # 比较需要两者
        "multi_hop_reasoning": "hybrid", # 多跳推理需要两者
        "temporal": "bm25",             # 时间相关 BM25 更精确
        "entity_centric": "bm25",       # 实体相关 BM25 更精确
    }

    return {
        "primary_intent": primary_intent,
        "all_intents": intents,
        "intent_scores": scores,
        "recommended_retrieval": recommended_retrieval.get(primary_intent, "hybrid"),
    }


# ============================================================
# 4. 数据质量过滤 (Data Quality Filtering)
# ============================================================

def check_data_quality(question: str, golden_answers: List[str]) -> Dict:
    """数据质量检查"""
    issues = []
    score = 1.0

    # 1. 问题长度检查
    q_len = len(question.split())
    if q_len < 3:
        issues.append("question_too_short")
        score -= 0.3
    elif q_len > 100:
        issues.append("question_too_long")
        score -= 0.1

    # 2. 问题是否是完整的问句
    if "?" not in question and not any(question.lower().startswith(w) for w in
                                        ["what", "who", "when", "where", "why", "how", "which", "name", "list", "tell", "describe"]):
        issues.append("not_a_question")
        score -= 0.2

    # 3. 答案完整性检查
    if not golden_answers:
        issues.append("no_golden_answer")
        score -= 0.5
    elif all(len(str(a).strip()) == 0 for a in golden_answers):
        issues.append("empty_golden_answer")
        score -= 0.5

    # 4. 检查问题-答案一致性（避免矛盾）
    for ans in golden_answers:
        ans_str = str(ans).lower().strip()
        if ans_str in question.lower():
            issues.append("answer_in_question")
            score -= 0.3
            break

    # 5. 特殊字符检查
    special_char_ratio = sum(1 for c in question if not c.isalnum() and c not in " ?!.,;:()-'\"") / max(len(question), 1)
    if special_char_ratio > 0.1:
        issues.append("excessive_special_chars")
        score -= 0.2

    return {
        "quality_score": max(score, 0.0),
        "is_acceptable": score >= 0.5,
        "issues": issues,
    }


# ============================================================
# 5. 对比增强 (Contrastive Augmentation)
# ============================================================

def generate_contrastive_pairs(
    question: str,
    correct_answer: str,
    distractors: List[str] = None,
    n_negatives: int = 2,
) -> List[Dict]:
    """
    构造对比学习数据对

    通过将正确答案和干扰答案配对，增强模型的判别能力
    """
    if not distractors:
        distractors = _generate_simple_distractors(question, correct_answer)

    pairs = []
    # 正例
    pairs.append({
        "question": question,
        "answer": correct_answer,
        "label": 1,  # positive
        "type": "positive",
    })

    # 负例
    for distractor in distractors[:n_negatives]:
        pairs.append({
            "question": question,
            "answer": distractor,
            "label": 0,  # negative
            "type": "negative",
        })

    return pairs


def _generate_simple_distractors(question: str, answer: str) -> List[str]:
    """生成简单干扰答案（基于规则）"""
    distractors = []

    # 规则1: 修改答案中的数字
    numbers = re.findall(r'\d+', str(answer))
    if numbers:
        n = int(numbers[0])
        distractors.append(str(answer).replace(numbers[0], str(n + 1)))
        distractors.append(str(answer).replace(numbers[0], str(max(0, n - 1))))

    # 规则2: 常见混淆实体
    common_confusions = {
        "paris": ["london", "berlin", "rome"],
        "london": ["paris", "manchester", "birmingham"],
        "china": ["japan", "korea", "india"],
        "english": ["french", "spanish", "german"],
        "python": ["java", "javascript", "ruby"],
    }
    ans_lower = str(answer).lower()
    for key, confusions in common_confusions.items():
        if key in ans_lower:
            for c in confusions:
                distractors.append(str(answer).lower().replace(key, c))
            break

    # 规则3: 翻转是否判断
    if str(answer).lower() in ["yes", "true", "correct"]:
        distractors.extend(["no", "false", "incorrect"])
    elif str(answer).lower() in ["no", "false", "incorrect"]:
        distractors.extend(["yes", "true", "correct"])

    return distractors[:3]


# ============================================================
# 6. 主处理流程
# ============================================================

def optimize_dataset(
    dataset: List[Dict],
    config: DataOptimizationConfig,
) -> List[Dict]:
    """
    数据集优化主函数

    对每条数据执行启用选项中指定的优化，返回增强后的数据集
    """
    print("=" * 60)
    print("  数据端优化处理")
    print("=" * 60)
    print(f"  输入样本数: {len(dataset)}")
    print(f"  启用优化: ", end="")
    enabled = []
    if config.enable_expansion:
        enabled.append("查询扩展")
    if config.enable_difficulty:
        enabled.append("难度分层")
    if config.enable_filtering:
        enabled.append("质量过滤")
    if config.enable_intent:
        enabled.append("意图分类")
    if config.enable_contrastive:
        enabled.append("对比增强")
    print(", ".join(enabled) if enabled else "无")
    print()

    # 提取原始字段
    questions = []
    golden_answers_list = []
    for item in dataset:
        if "prompt" in item and isinstance(item["prompt"], list):
            # 从 parquet 格式中提取
            for p in item["prompt"]:
                if p.get("role") == "user":
                    # 从 prompt 模板中提取原始问题
                    content = p.get("content", "")
                    q = _extract_question_from_prompt(content)
                    questions.append(q)
                    break
            else:
                questions.append("")
        elif "question" in item:
            questions.append(item["question"])
        else:
            questions.append("")

        if "reward_model" in item:
            answers = item.get("reward_model", {}).get("ground_truth", {}).get("target", [])
        elif "golden_answers" in item:
            answers = item["golden_answers"]
        else:
            answers = []
        golden_answers_list.append(answers if isinstance(answers, list) else [answers])

    # 优化计数
    stats = {
        "total_input": len(dataset),
        "expanded_queries": 0,
        "filtered_out": 0,
        "difficulty_easy": 0,
        "difficulty_medium": 0,
        "difficulty_hard": 0,
        "total_output": len(dataset),
    }

    optimized = []

    for i, item in enumerate(dataset):
        q = questions[i]
        answers = golden_answers_list[i]

        # 4. 质量过滤
        if config.enable_filtering:
            quality = check_data_quality(q, answers)
            if not quality["is_acceptable"]:
                stats["filtered_out"] += 1
                continue

        # 3. 查询意图分类
        if config.enable_intent and q:
            intent = classify_query_intent(q)
            item["query_intent"] = intent

        # 2. 难度分层
        if config.enable_difficulty and q:
            difficulty = estimate_retrieval_difficulty(q, answers)
            item["difficulty"] = difficulty
            stats[f"difficulty_{difficulty['difficulty_level']}"] += 1

        # 优化后的数据项
        optimized.append(item)

        # 5. 对比增强
        if config.enable_contrastive and answers:
            contrastive_pairs = generate_contrastive_pairs(
                q, str(answers[0]) if answers else "",
                n_negatives=config.contrastive_negative_count,
            )
            if "contrastive_pairs" not in item:
                item["contrastive_pairs"] = contrastive_pairs

        # 1. 查询扩展 - 最后处理（可能创建新样本）
        if config.enable_expansion and q:
            expanded = expand_queries([q], config)[0]
            for variant in expanded:
                variant_item = dict(item)
                if "prompt" in variant_item and isinstance(variant_item["prompt"], list):
                    for p in variant_item["prompt"]:
                        if p.get("role") == "user":
                            p["content"] = p["content"].replace(q, variant)
                elif "question" in variant_item:
                    variant_item["question"] = variant
                variant_item["_is_expanded"] = True
                variant_item["_original_question"] = q
                optimized.append(variant_item)
                stats["expanded_queries"] += 1

    stats["total_output"] = len(optimized)

    # 打印统计
    print("  优化统计:")
    print(f"    输入: {stats['total_input']} 条")
    if config.enable_filtering:
        print(f"    过滤: {stats['filtered_out']} 条")
    if config.enable_expansion:
        print(f"    扩展: +{stats['expanded_queries']} 条")
    print(f"    输出: {stats['total_output']} 条")
    if config.enable_difficulty:
        print(f"    难度分布: easy={stats['difficulty_easy']}, "
              f"medium={stats['difficulty_medium']}, hard={stats['difficulty_hard']}")

    print(f"\n  {'✓' if stats['total_output'] > 0 else '✗'} 优化完成")

    return optimized


def _extract_question_from_prompt(content: str) -> str:
    """从 Search-R1 的 prompt 模板中提取原始问题"""
    # 模板: "...Question: {question}\n"
    m = re.search(r'Question:\s*(.+?)(?:\n|$)', content, re.DOTALL)
    if m:
        q = m.group(1).strip()
        # 去掉末尾的句号（如果有问号则保留）
        if q.endswith('.') and '?' not in q:
            q = q[:-1] + '?'
        return q
    return content


# ============================================================
# 7. 演示模式
# ============================================================

def demo_data_optimizations():
    """演示所有数据优化策略的效果"""
    print("=" * 60)
    print("  Search-R1 数据端优化策略演示")
    print("=" * 60)

    demo_queries = [
        ("What is the capital of France?", ["Paris"]),
        ("When was the iPhone first released and who created it?", ["2007", "Steve Jobs"]),
        ("Compare the Python programming language with Java", ["Python is dynamically typed", "Java is statically typed"]),
        ("What is the difference between mitosis and meiosis?", ["Mitosis produces two identical daughter cells", "Meiosis produces four genetically different cells"]),
        ("Name the largest planet in our solar system", ["Jupiter"]),
    ]

    # 演示1: 查询扩展
    print("\n--- 1. 查询扩展示例 ---")
    for q, _ in demo_queries[:2]:
        variants = expand_queries([q], DataOptimizationConfig(
            enable_expansion=True,
            expansion_methods=["synonym", "back_translate_like", "question_rephrase"],
            expansion_factor=2,
        ))[0]
        print(f"\n  原始: {q}")
        for v in variants:
            print(f"  变体: {v}")

    # 演示2: 难度分层
    print("\n\n--- 2. 难度分层示例 ---")
    for q, answers in demo_queries:
        diff = estimate_retrieval_difficulty(q, answers)
        print(f"  [{diff['difficulty_level']:6s}] complexity={diff['complexity']:.2f} "
              f"retrieval_dep={diff['retrieval_dependency']:.2f} | {q[:60]}...")

    # 演示3: 意图分类
    print("\n\n--- 3. 查询意图分类示例 ---")
    for q, _ in demo_queries:
        intent = classify_query_intent(q)
        print(f"  [{intent['primary_intent']:<20s}] → 推荐检索: {intent['recommended_retrieval']:<7s} | {q[:60]}...")

    # 演示4: 质量检查
    print("\n\n--- 4. 数据质量过滤示例 ---")
    bad_examples = [
        ("x y z", ["test"]),      # question_too_short
        ("", ["answer"]),          # empty question
        ("What is Paris?", []),    # no answer
        ("What is Paris?", ["Paris"]),  # answer in question
    ]
    for q, answers in bad_examples + [demo_queries[0]]:
        quality = check_data_quality(q, answers)
        status = "[OK]" if quality["is_acceptable"] else "[FILTERED]"
        print(f"  {status} score={quality['quality_score']:.2f} issues={quality['issues']} | Q: {q}")

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)
    print("""
  数据端优化总结:

  1. 查询扩展 - 为每条训练数据生成2-3个语义等价变体，增加多样性
     → 预期收益: 模型泛化能力提升，减少对特定问法的过拟合

  2. 难度分层 - 按检索依赖度将数据分为 easy/medium/hard 三级
     → 预期收益: 支持课程学习，从简单到困难逐步训练

  3. 查询意图分类 - 自动识别查询类型，推荐最优检索策略
     → 预期收益: 可针对不同意图使用不同检索器组合

  4. 数据质量过滤 - 识别低质量样本并过滤
     → 预期收益: 减少噪声数据对训练的影响

  5. 对比增强 - 构造正负样本对，增强判别能力
     → 预期收益: 模型学会区分正确答案和干扰答案
  """)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Search-R1 数据端优化")

    parser.add_argument("--input_path", type=str, default=None,
                        help="输入数据路径 (.parquet 或 .jsonl)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出数据路径")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="运行演示模式（默认）")

    # 优化开关
    parser.add_argument("--enable_expansion", action="store_true",
                        help="启用查询扩展")
    parser.add_argument("--enable_difficulty", action="store_true",
                        help="启用难度分层")
    parser.add_argument("--enable_filtering", action="store_true",
                        help="启用数据质量过滤")
    parser.add_argument("--enable_intent", action="store_true",
                        help="启用查询意图分类")
    parser.add_argument("--enable_contrastive", action="store_true",
                        help="启用对比增强")

    args = parser.parse_args()

    if args.input_path and args.output_path:
        # 实际处理模式
        config = DataOptimizationConfig(
            enable_expansion=args.enable_expansion,
            enable_difficulty=args.enable_difficulty,
            enable_filtering=args.enable_filtering,
            enable_intent=args.enable_intent,
            enable_contrastive=args.enable_contrastive,
        )

        if args.input_path.endswith('.parquet'):
            from datasets import load_dataset
            dataset = load_dataset('parquet', data_files=args.input_path, split='train')
            dataset = [dict(item) for item in dataset]
        elif args.input_path.endswith('.jsonl'):
            dataset = []
            with open(args.input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataset.append(json.loads(line))
        else:
            raise ValueError(f"不支持的文件格式: {args.input_path}")

        optimized = optimize_dataset(dataset, config)

        if args.output_path.endswith('.parquet'):
            from datasets import Dataset
            ds = Dataset.from_list(optimized)
            ds.to_parquet(args.output_path)
        elif args.output_path.endswith('.jsonl'):
            with open(args.output_path, 'w', encoding='utf-8') as f:
                for item in optimized:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"  输出已保存: {args.output_path}")
    else:
        # 演示模式
        demo_data_optimizations()


if __name__ == "__main__":
    main()
