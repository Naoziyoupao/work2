#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 Magpie 数据集

将 Magpie 数据集转换为统一的问答格式，用作合成数据源

功能：
1. 读取原始 Magpie 数据
2. 提取指令-响应对
3. 转换为与 OASST2 相同的统一格式
4. 划分训练集/验证集
5. 计算统计信息
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
    # Magpie 数据集通常包含 conversations 字段
    # 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    if 'conversations' in item:
        conversations = item['conversations']
        if len(conversations) >= 2:
            # 取第一轮对话
            if conversations[0].get('role') in ['user', 'human'] and \
               conversations[1].get('role') in ['assistant', 'gpt']:
                return {
                    'instruction': conversations[0].get('content', '').strip(),
                    'response': conversations[1].get('content', '').strip()
                }
    
    # 其他可能的字段格式
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
    val_ratio: float = 0.1,
    seed: int = RANDOM_SEED,
    max_samples: int = None
):
    """
    处理 Magpie 数据集
    
    Args:
        input_dir: 输入目录（原始数据）
        output_dir: 输出目录
        val_ratio: 验证集比例
        seed: 随机种子
        max_samples: 最大样本数（用于快速测试）
    """
    print("="*80)
    print("🚀 开始处理 Magpie 数据集")
    print("="*80)
    print(f"📂 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"⚖️  验证集比例: {val_ratio}")
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
    print("\n🔄 提取指令-响应对...")
    processed_data = []
    skipped_count = 0
    
    for idx, item in enumerate(tqdm(raw_data, desc="处理数据")):
        # 限制样本数（用于测试）
        if max_samples and len(processed_data) >= max_samples:
            break
        
        # 提取指令和响应
        extracted = extract_instruction_response(item)
        
        if extracted and extracted['instruction'] and extracted['response']:
            # 创建统一格式
            processed_item = {
                'conversation_id': f"magpie_{idx}",
                'text': f"Question: {extracted['instruction']}\nAnswer: {extracted['response']}",
                'instruction': extracted['instruction'],
                'response': extracted['response']
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
    
    # 划分训练集和验证集
    print(f"\n⚖️  划分训练集/验证集 ({(1-val_ratio)*100:.0f}:{val_ratio*100:.0f})...")
    train_data, val_data = train_test_split(
        processed_data,
        test_size=val_ratio,
        random_state=seed
    )
    
    print(f"✅ 划分完成:")
    print(f"   - 训练集: {len(train_data)} 条")
    print(f"   - 验证集: {len(val_data)} 条")
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'synthetic_train.jsonl')
    val_file = os.path.join(output_dir, 'synthetic_val.jsonl')
    
    print(f"\n💾 保存数据...")
    
    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 训练集已保存: {train_file}")
    
    # 保存验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 验证集已保存: {val_file}")
    
    # 计算统计信息
    print(f"\n📊 计算统计信息...")
    
    train_instruction_lengths = [len(item['instruction']) for item in train_data]
    train_response_lengths = [len(item['response']) for item in train_data]
    val_instruction_lengths = [len(item['instruction']) for item in val_data]
    val_response_lengths = [len(item['response']) for item in val_data]
    
    stats = {
        "dataset": "Magpie-Align/Magpie-Pro-300K-Filtered",
        "total_samples": len(processed_data),
        "train": {
            "count": len(train_data),
            "instruction_length": {
                "mean": float(np.mean(train_instruction_lengths)),
                "median": float(np.median(train_instruction_lengths)),
                "std": float(np.std(train_instruction_lengths)),
                "min": int(np.min(train_instruction_lengths)),
                "max": int(np.max(train_instruction_lengths))
            },
            "response_length": {
                "mean": float(np.mean(train_response_lengths)),
                "median": float(np.median(train_response_lengths)),
                "std": float(np.std(train_response_lengths)),
                "min": int(np.min(train_response_lengths)),
                "max": int(np.max(train_response_lengths))
            }
        },
        "val": {
            "count": len(val_data),
            "instruction_length": {
                "mean": float(np.mean(val_instruction_lengths)),
                "median": float(np.median(val_instruction_lengths)),
                "std": float(np.std(val_instruction_lengths)),
                "min": int(np.min(val_instruction_lengths)),
                "max": int(np.max(val_instruction_lengths))
            },
            "response_length": {
                "mean": float(np.mean(val_response_lengths)),
                "median": float(np.median(val_response_lengths)),
                "std": float(np.std(val_response_lengths)),
                "min": int(np.min(val_response_lengths)),
                "max": int(np.max(val_response_lengths))
            }
        }
    }
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'magpie_data_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ 统计信息已保存: {stats_file}")
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("📈 数据统计摘要")
    print("="*80)
    print(f"训练集: {stats['train']['count']} 条")
    print(f"  指令长度: {stats['train']['instruction_length']['mean']:.1f} ± {stats['train']['instruction_length']['std']:.1f} 字符")
    print(f"  响应长度: {stats['train']['response_length']['mean']:.1f} ± {stats['train']['response_length']['std']:.1f} 字符")
    print(f"\n验证集: {stats['val']['count']} 条")
    print(f"  指令长度: {stats['val']['instruction_length']['mean']:.1f} ± {stats['val']['instruction_length']['std']:.1f} 字符")
    print(f"  响应长度: {stats['val']['response_length']['mean']:.1f} ± {stats['val']['response_length']['std']:.1f} 字符")
    
    print("\n" + "="*80)
    print("🎉 Magpie 数据集处理完成！")
    print("="*80)
    print(f"📁 输出目录: {output_dir}")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {stats_file}")
    print("\n💡 下一步:")
    print("   运行分类数据准备脚本，使用 Magpie 作为合成数据源:")
    print("   python scripts/data/prepare_classification_data.py \\")
    print("       --real_train data/processed/oasst2/first_turn_train_5k.jsonl \\")
    print("       --real_val data/processed/oasst2/first_turn_val_5k.jsonl \\")
    print("       --synthetic_train data/processed/magpie/synthetic_train.jsonl \\")
    print("       --synthetic_val data/processed/magpie/synthetic_val.jsonl \\")
    print("       --output_dir data/processed/magpie_classification")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="处理 Magpie 数据集，转换为统一格式"
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
        '--val_ratio',
        type=float,
        default=0.1,
        help='验证集比例 (默认: 0.1)'
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
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
