#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类数据准备脚本

合并真实数据和合成数据，准备用于训练二分类器（区分真实vs合成）

功能：
1. 读取真实数据（OASST2第一轮对话）
2. 读取合成数据（Qwen3-32B生成）
3. 添加label字段（0=真实，1=合成）
4. 打乱数据顺序
5. 划分训练集和验证集
6. 保存为统一格式

输出格式：
{
    "conversation_id": "xxx",
    "text": "问题: ...\n答案: ...",
    "instruction": "...",
    "response": "...",
    "label": 0 or 1  # 0=真实，1=合成
}
"""

import os
import json
import argparse
import random
from typing import List, Dict
from collections import defaultdict

# 默认路径
DEFAULT_REAL_TRAIN = "data/processed/oasst2/first_turn_train_5k.jsonl"
DEFAULT_REAL_VAL = "data/processed/oasst2/first_turn_val_5k.jsonl"
DEFAULT_SYNTHETIC_TRAIN = "data/processed/oasst2/synthetic_train_5k.jsonl"
DEFAULT_OUTPUT_DIR = "data/processed/oasst2/classification"

# 随机种子
RANDOM_SEED = 42


def load_and_label_data(file_path: str, label: int, data_type: str = "train") -> List[Dict]:
    """
    加载数据并添加label字段
    
    Args:
        file_path: 数据文件路径
        label: 标签（0=真实，1=合成）
        data_type: 数据类型（"train" 或 "val"）
    
    Returns:
        带标签的数据列表
    """
    print(f"📥 加载{data_type}数据: {file_path}")
    
    labeled_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # 统一数据格式 (使用英文格式)
            classification_item = {
                "conversation_id": item.get('conversation_id', ''),
                "text": f"Question: {item['instruction']}\nAnswer: {item['response']}",
                "instruction": item['instruction'],
                "response": item['response'],
                "label": label
            }
            
            # 保留一些元数据（可选）
            if 'response_tokens' in item:
                classification_item['response_tokens'] = item['response_tokens']
            if 'response_length' in item:
                classification_item['response_length'] = item['response_length']
            
            # 如果是合成数据，保留额外信息
            if label == 1:
                if 'rm_score' in item:
                    classification_item['rm_score'] = item['rm_score']
                if 'final_score' in item:
                    classification_item['final_score'] = item['final_score']
                if 'temperature' in item:
                    classification_item['temperature'] = item['temperature']
            
            labeled_data.append(classification_item)
    
    print(f"✅ 加载 {len(labeled_data)} 条数据 (label={label})")
    return labeled_data


def balance_data(real_data: List[Dict], synthetic_data: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """
    平衡两类数据的数量（确保分类器训练平衡）
    
    Args:
        real_data: 真实数据列表
        synthetic_data: 合成数据列表
    
    Returns:
        平衡后的(real_data, synthetic_data)
    """
    real_count = len(real_data)
    synthetic_count = len(synthetic_data)
    
    print(f"\n📊 数据平衡检查:")
    print(f"  真实数据: {real_count} 条")
    print(f"  合成数据: {synthetic_count} 条")
    
    if real_count == synthetic_count:
        print(f"✅ 数据已平衡")
        return real_data, synthetic_data
    
    # 如果数量不平衡，下采样较多的那一类
    if real_count > synthetic_count:
        print(f"⚖️  下采样真实数据至 {synthetic_count} 条")
        real_data = random.sample(real_data, synthetic_count)
    else:
        print(f"⚖️  下采样合成数据至 {real_count} 条")
        synthetic_data = random.sample(synthetic_data, real_count)
    
    return real_data, synthetic_data


def shuffle_and_save(data: List[Dict], output_file: str):
    """
    打乱数据并保存
    
    Args:
        data: 数据列表
        output_file: 输出文件路径
    """
    # 打乱数据
    random.shuffle(data)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(data)} 条数据到: {output_file}")


def generate_statistics(train_file: str, val_file: str, output_dir: str):
    """
    生成数据集统计信息
    
    Args:
        train_file: 训练集文件路径
        val_file: 验证集文件路径
        output_dir: 输出目录
    """
    print(f"\n📊 生成统计信息...")
    
    # 统计训练集
    train_stats = defaultdict(int)
    train_response_lengths = {'real': [], 'synthetic': []}
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            label = item['label']
            train_stats[f'label_{label}'] += 1
            
            if 'response_tokens' in item:
                if label == 0:
                    train_response_lengths['real'].append(item['response_tokens'])
                else:
                    train_response_lengths['synthetic'].append(item['response_tokens'])
    
    # 统计验证集
    val_stats = defaultdict(int)
    val_response_lengths = {'real': [], 'synthetic': []}
    
    if os.path.exists(val_file):
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                label = item['label']
                val_stats[f'label_{label}'] += 1
                
                if 'response_tokens' in item:
                    if label == 0:
                        val_response_lengths['real'].append(item['response_tokens'])
                    else:
                        val_response_lengths['synthetic'].append(item['response_tokens'])
    
    # 构建统计信息
    import numpy as np
    
    stats = {
        "train": {
            "total": train_stats['label_0'] + train_stats['label_1'],
            "real_count": train_stats['label_0'],
            "synthetic_count": train_stats['label_1'],
            "balance_ratio": train_stats['label_0'] / (train_stats['label_1'] + 1e-10)
        },
        "val": {
            "total": val_stats['label_0'] + val_stats['label_1'],
            "real_count": val_stats['label_0'],
            "synthetic_count": val_stats['label_1'],
            "balance_ratio": val_stats['label_0'] / (val_stats['label_1'] + 1e-10)
        }
    }
    
    # 添加长度统计
    if train_response_lengths['real']:
        stats['train']['real_length_stats'] = {
            "mean": float(np.mean(train_response_lengths['real'])),
            "median": float(np.median(train_response_lengths['real'])),
            "std": float(np.std(train_response_lengths['real']))
        }
    
    if train_response_lengths['synthetic']:
        stats['train']['synthetic_length_stats'] = {
            "mean": float(np.mean(train_response_lengths['synthetic'])),
            "median": float(np.median(train_response_lengths['synthetic'])),
            "std": float(np.std(train_response_lengths['synthetic']))
        }
    
    if val_response_lengths['real']:
        stats['val']['real_length_stats'] = {
            "mean": float(np.mean(val_response_lengths['real'])),
            "median": float(np.median(val_response_lengths['real'])),
            "std": float(np.std(val_response_lengths['real']))
        }
    
    if val_response_lengths['synthetic']:
        stats['val']['synthetic_length_stats'] = {
            "mean": float(np.mean(val_response_lengths['synthetic'])),
            "median": float(np.median(val_response_lengths['synthetic'])),
            "std": float(np.std(val_response_lengths['synthetic']))
        }
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'classification_data_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 统计信息已保存: {stats_file}")
    
    # 打印关键统计
    print(f"\n📈 数据集统计:")
    print(f"  训练集: {stats['train']['total']} 条")
    print(f"    - 真实数据: {stats['train']['real_count']} 条")
    print(f"    - 合成数据: {stats['train']['synthetic_count']} 条")
    print(f"    - 平衡比: {stats['train']['balance_ratio']:.2f}")
    
    if stats['val']['total'] > 0:
        print(f"  验证集: {stats['val']['total']} 条")
        print(f"    - 真实数据: {stats['val']['real_count']} 条")
        print(f"    - 合成数据: {stats['val']['synthetic_count']} 条")
        print(f"    - 平衡比: {stats['val']['balance_ratio']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="准备分类器训练数据 - 合并真实数据和合成数据"
    )
    
    parser.add_argument(
        '--real_train',
        type=str,
        default=DEFAULT_REAL_TRAIN,
        help='真实数据训练集路径'
    )
    parser.add_argument(
        '--real_val',
        type=str,
        default=DEFAULT_REAL_VAL,
        help='真实数据验证集路径'
    )
    parser.add_argument(
        '--synthetic_train',
        type=str,
        default=DEFAULT_SYNTHETIC_TRAIN,
        help='合成数据训练集路径'
    )
    parser.add_argument(
        '--synthetic_val',
        type=str,
        default=None,
        help='合成数据验证集路径（可选）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='输出目录'
    )
    parser.add_argument(
        '--no_balance',
        action='store_true',
        help='不进行数据平衡（默认会平衡两类数据数量）'
    )
    parser.add_argument(
        '--create_synthetic_val',
        action='store_true',
        help='从合成训练集中划分验证集（如果没有单独的合成验证集）'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.1,
        help='验证集比例（仅在create_synthetic_val时使用）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("="*80)
    print("🎯 准备分类器训练数据")
    print("="*80)
    
    # 加载真实数据
    print("\n### 加载真实数据 ###")
    real_train_data = load_and_label_data(args.real_train, label=0, data_type="train")
    
    real_val_data = []
    if os.path.exists(args.real_val):
        real_val_data = load_and_label_data(args.real_val, label=0, data_type="val")
    else:
        print(f"⚠️  未找到真实数据验证集: {args.real_val}")
    
    # 加载合成数据
    print("\n### 加载合成数据 ###")
    synthetic_train_data = load_and_label_data(args.synthetic_train, label=1, data_type="train")
    
    # 处理合成数据验证集
    synthetic_val_data = []
    if args.synthetic_val and os.path.exists(args.synthetic_val):
        # 直接加载提供的合成验证集
        synthetic_val_data = load_and_label_data(args.synthetic_val, label=1, data_type="val")
    elif args.create_synthetic_val:
        # 从训练集中划分验证集
        val_size = int(len(synthetic_train_data) * args.val_ratio)
        random.shuffle(synthetic_train_data)
        synthetic_val_data = synthetic_train_data[:val_size]
        synthetic_train_data = synthetic_train_data[val_size:]
        print(f"✅ 从合成训练集中划分出 {len(synthetic_val_data)} 条验证数据")
    
    # 数据平衡
    if not args.no_balance:
        print("\n### 数据平衡 ###")
        real_train_data, synthetic_train_data = balance_data(real_train_data, synthetic_train_data)
        
        if real_val_data and synthetic_val_data:
            real_val_data, synthetic_val_data = balance_data(real_val_data, synthetic_val_data)
    
    # 合并训练集
    print("\n### 合并并保存数据 ###")
    train_data = real_train_data + synthetic_train_data
    train_output = os.path.join(args.output_dir, 'train.jsonl')
    shuffle_and_save(train_data, train_output)
    
    # 合并验证集
    if real_val_data and synthetic_val_data:
        val_data = real_val_data + synthetic_val_data
        val_output = os.path.join(args.output_dir, 'val.jsonl')
        shuffle_and_save(val_data, val_output)
    elif real_val_data:
        # 只有真实验证集，从训练集中抽取合成验证集
        print(f"⚠️  没有合成验证集，从训练集中抽取...")
        synthetic_val_size = len(real_val_data)
        synthetic_val_data = synthetic_train_data[:synthetic_val_size]
        synthetic_train_data = synthetic_train_data[synthetic_val_size:]
        
        # 重新保存训练集
        train_data = real_train_data + synthetic_train_data
        shuffle_and_save(train_data, train_output)
        
        # 保存验证集
        val_data = real_val_data + synthetic_val_data
        val_output = os.path.join(args.output_dir, 'val.jsonl')
        shuffle_and_save(val_data, val_output)
    else:
        val_output = None
        print(f"⚠️  未创建验证集")
    
    # 生成统计信息
    if val_output:
        generate_statistics(train_output, val_output, args.output_dir)
    else:
        print(f"\n⚠️  跳过统计信息生成（缺少验证集）")
    
    # 打印输出路径
    print("\n" + "="*80)
    print("🎉 分类数据准备完成！")
    print("="*80)
    print(f"📁 输出目录: {args.output_dir}")
    print(f"  训练集: {train_output}")
    if val_output:
        print(f"  验证集: {val_output}")
    print("="*80)


if __name__ == "__main__":
    main()