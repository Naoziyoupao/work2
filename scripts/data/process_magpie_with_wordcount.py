#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 Magpie 数据集 - 添加单词计数

将 Magpie 数据集转换为统一的问答格式，并计算单词数
"""

import os
import json
import argparse
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

# 默认配置
DEFAULT_INPUT_DIR = "data/raw/magpie"
DEFAULT_OUTPUT_DIR = "data/processed/magpie"
RANDOM_SEED = 42


def extract_instruction_response(item: dict) -> dict:
    """
    从 Magpie 数据项中提取指令和响应
    
    Args:
        item: Magpie 数据项
    
    Returns:
        包含 instruction 和 response 的字典，如果提取失败则返回 None
    """
    if 'conversations' in item:
        conversations = item['conversations']
        if len(conversations) >= 2:
            # Magpie 使用 'from' 和 'value'
            first = conversations[0]
            second = conversations[1]
            
            # 支持多种字段名
            first_role = first.get('from') or first.get('role')
            first_content = first.get('value') or first.get('content')
            second_role = second.get('from') or second.get('role')
            second_content = second.get('value') or second.get('content')
            
            if first_role in ['user', 'human'] and \
               second_role in ['assistant', 'gpt'] and \
               first_content and second_content:
                return {
                    'instruction': first_content.strip(),
                    'response': second_content.strip()
                }
    
    elif 'instruction' in item and 'response' in item:
        return {
            'instruction': item['instruction'].strip(),
            'response': item['response'].strip()
        }
    
    elif 'prompt' in item and 'completion' in item:
        return {
            'instruction': item['prompt'].strip(),
            'response': item['completion'].strip()
        }
    
    elif 'input' in item and 'output' in item:
        return {
            'instruction': item['input'].strip(),
            'response': item['output'].strip()
        }
    
    return None


def process_magpie(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = RANDOM_SEED,
    max_samples: int = None
):
    """
    处理 Magpie 数据集
    
    Args:
        input_dir: 输入目录（原始数据）
        output_dir: 输出目录
        seed: 随机种子
        max_samples: 最大样本数（用于快速测试）
    """
    print("="*80)
    print("🚀 开始处理 Magpie 数据集（添加单词计数）")
    print("="*80)
    print(f"📂 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎲 随机种子: {seed}")
    if max_samples:
        print(f"📊 最大样本数: {max_samples}")
    print("="*80)
    
    # 加载数据集
    print("\n📥 加载 Magpie 数据集...")
    try:
        dataset = load_from_disk(input_dir)
        
        # 获取训练集数据
        if 'train' in dataset:
            raw_data = dataset['train']
        else:
            # 如果没有 train split，取第一个 split
            first_split = list(dataset.keys())[0]
            raw_data = dataset[first_split]
            print(f"⚠️  未找到 'train' split，使用 '{first_split}' split")
        
        print(f"✅ 加载完成，共 {len(raw_data)} 条原始数据")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(f"💡 提示: 请先运行 python scripts/data/download_magpie.py")
        raise
    
    # 提取指令-响应对
    print("\n🔄 提取指令-响应对并计算单词数...")
    processed_data = []
    skipped_count = 0
    
    for idx, item in enumerate(tqdm(raw_data, desc="处理数据")):
        # 限制样本数（用于测试）
        if max_samples and len(processed_data) >= max_samples:
            break
        
        # 提取指令和响应
        extracted = extract_instruction_response(item)
        
        if extracted and extracted['instruction'] and extracted['response']:
            instruction = extracted['instruction']
            response = extracted['response']
            
            # 计算单词数
            instruction_word_count = len(instruction.split())
            response_word_count = len(response.split())
            
            # 创建统一格式
            processed_item = {
                'conversation_id': f"magpie_{idx}",
                'instruction': instruction,
                'response': response,
                'instruction_word_count': instruction_word_count,
                'response_word_count': response_word_count
            }
            
            # 添加可选元数据
            if 'quality_score' in item:
                processed_item['quality_score'] = item['quality_score']
            if 'model' in item:
                processed_item['source_model'] = item['model']
            
            processed_data.append(processed_item)
        else:
            skipped_count += 1
    
    print(f"✅ 提取完成:")
    print(f"   - 成功: {len(processed_data)} 条")
    print(f"   - 跳过: {skipped_count} 条")
    
    if len(processed_data) == 0:
        print("\n❌ 错误: 没有成功提取任何数据")
        print("💡 提示: 检查数据格式或查看第一条样例:")
        print(json.dumps(raw_data[0], ensure_ascii=False, indent=2))
        return
    
    # 保存全部数据（不划分）
    os.makedirs(output_dir, exist_ok=True)
    
    all_file = os.path.join(output_dir, 'synthetic_all.jsonl')
    
    print(f"\n💾 保存数据...")
    
    # 保存全部数据
    with open(all_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 全部数据已保存: {all_file}")
    
    # 计算统计信息
    print(f"\n📊 计算统计信息...")
    
    instruction_word_counts = [item['instruction_word_count'] for item in processed_data]
    response_word_counts = [item['response_word_count'] for item in processed_data]
    
    stats = {
        "dataset": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "total_samples": len(processed_data),
        "instruction_word_count": {
            "mean": float(np.mean(instruction_word_counts)),
            "median": float(np.median(instruction_word_counts)),
            "std": float(np.std(instruction_word_counts)),
            "min": int(np.min(instruction_word_counts)),
            "max": int(np.max(instruction_word_counts)),
            "percentiles": {
                "25": float(np.percentile(instruction_word_counts, 25)),
                "50": float(np.percentile(instruction_word_counts, 50)),
                "75": float(np.percentile(instruction_word_counts, 75)),
                "90": float(np.percentile(instruction_word_counts, 90)),
                "95": float(np.percentile(instruction_word_counts, 95))
            }
        },
        "response_word_count": {
            "mean": float(np.mean(response_word_counts)),
            "median": float(np.median(response_word_counts)),
            "std": float(np.std(response_word_counts)),
            "min": int(np.min(response_word_counts)),
            "max": int(np.max(response_word_counts)),
            "percentiles": {
                "25": float(np.percentile(response_word_counts, 25)),
                "50": float(np.percentile(response_word_counts, 50)),
                "75": float(np.percentile(response_word_counts, 75)),
                "90": float(np.percentile(response_word_counts, 90)),
                "95": float(np.percentile(response_word_counts, 95))
            }
        }
    }
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'synthetic_all_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ 统计信息已保存: {stats_file}")
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("📈 数据统计摘要")
    print("="*80)
    print(f"总样本数: {stats['total_samples']} 条")
    print(f"  指令单词数: {stats['instruction_word_count']['mean']:.1f} ± {stats['instruction_word_count']['std']:.1f}")
    print(f"    中位数: {stats['instruction_word_count']['median']:.1f}")
    print(f"    范围: [{stats['instruction_word_count']['min']}, {stats['instruction_word_count']['max']}]")
    print(f"  响应单词数: {stats['response_word_count']['mean']:.1f} ± {stats['response_word_count']['std']:.1f}")
    print(f"    中位数: {stats['response_word_count']['median']:.1f}")
    print(f"    范围: [{stats['response_word_count']['min']}, {stats['response_word_count']['max']}]")
    
    print("\n" + "="*80)
    print("🎉 Magpie 数据集处理完成！")
    print("="*80)
    print(f"📁 输出目录: {output_dir}")
    print(f"  - {all_file}")
    print(f"  - {stats_file}")
    print("\n💡 下一步:")
    print("   运行长度分布匹配脚本，根据 OASST2 分布采样 Magpie:")
    print("   python scripts/data/match_and_merge_datasets.py")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="处理 Magpie 数据集，添加单词计数"
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f'输入目录（原始 Magpie 数据）(默认: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'随机种子 (默认: {RANDOM_SEED})'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大样本数（用于快速测试）'
    )
    
    args = parser.parse_args()
    
    process_magpie(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
