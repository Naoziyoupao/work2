#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机采样并合并数据集 (去重、防泄露版本)

功能：
1. 加载 OASST2 真实多轮对话数据
2. 加载 Magpie 合成数据并进行严格去重
3. 根据 OASST2 的数据量，从去重后的 Magpie 中按 1:1 比例纯随机抽取
4. 使用“全局洗牌+顺序切片”机制，绝对防止训练集与测试集之间的数据泄露
5. 合并并添加标签 (real=0, synthetic=1)
6. 生成统计报告
"""

import os
import json
import argparse
import hashlib
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# 默认配置
DEFAULT_OASST2_TRAIN = "data/processed/oasst2/real_multiturn_train.jsonl"
DEFAULT_OASST2_TEST = "data/processed/oasst2/real_multiturn_test.jsonl"
DEFAULT_MAGPIE_ALL = "data/processed/magpie/synthetic_all.jsonl"
DEFAULT_OUTPUT_DIR = "data/processed/classification_random"
RANDOM_SEED = 42


def load_jsonl(filepath: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], filepath: str):
    """保存为 JSONL 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def deduplicate_data(data: List[Dict]) -> List[Dict]:
    """
    使用 MD5 哈希进行严格去重
    防止 Magpie 中出现完全相同的对话内容
    """
    seen_hashes = set()
    deduped_data = []
    
    for item in tqdm(data, desc=" 去重处理中"):
        # 移除可能导致误判的动态字段（比如之前加过的标签或无关的ID）
        # 提取核心内容进行对比，这里简单粗暴地将特定核心字段或整个字典转化为排序后的字符串
        core_content = {k: v for k, v in item.items() if k not in ['id', 'label', 'source']}
        content_str = json.dumps(core_content, sort_keys=True, ensure_ascii=False)
        
        # 计算 MD5
        content_hash = hashlib.md5(content_str.encode('utf-8')).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            deduped_data.append(item)
            
    return deduped_data


def sample_and_merge(
    oasst2_train_file: str,
    oasst2_test_file: str,
    magpie_all_file: str,
    output_dir: str,
    seed: int = RANDOM_SEED
):
    """
    随机抽取并合并数据集，严格防止数据泄露
    """
    print("="*80)
    print("🚀 开始随机抽取并合并数据集 (严格去重+防泄露)")
    print("="*80)
    print(f"📂 OASST2 训练集: {oasst2_train_file}")
    print(f"📂 OASST2 测试集: {oasst2_test_file}")
    print(f"📂 Magpie 全部数据: {magpie_all_file}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎲 随机种子: {seed}")
    print("="*80)
    
    # 1. 加载数据
    print("\n📥 加载数据...")
    oasst2_train = load_jsonl(oasst2_train_file)
    oasst2_test = load_jsonl(oasst2_test_file)
    magpie_all = load_jsonl(magpie_all_file)
    
    num_train_needed = len(oasst2_train)
    num_test_needed = len(oasst2_test)
    
    print(f"✅ OASST2 训练集: {num_train_needed} 条")
    print(f"✅ OASST2 测试集: {num_test_needed} 条")
    print(f"✅ Magpie 初始加载: {len(magpie_all)} 条")
    
    # 2. 严格去重 Magpie 数据
    print("\n🧹 正在对 Magpie 合成数据进行去重...")
    magpie_deduped = deduplicate_data(magpie_all)
    print(f"✅ 去重完成: 剔除了 {len(magpie_all) - len(magpie_deduped)} 条重复数据，剩余 {len(magpie_deduped)} 条有效数据")
    
    # 检查余量是否充足
    if len(magpie_deduped) < num_train_needed + num_test_needed:
        raise ValueError(f"❌ 严重错误: Magpie 去重后剩余数据 ({len(magpie_deduped)}) 不足，无法满足 1:1 匹配所需总量 ({num_train_needed + num_test_needed})")

    # 3. 全局洗牌 + 顺序切片 (核心防泄露机制)
    print("\n🔀 进行全局洗牌与物理隔离切片 (防止数据泄露)...")
    np.random.seed(seed)
    # 将 list 转换为 numpy array 方便高级索引，或者直接使用 np.random.shuffle
    np.random.shuffle(magpie_deduped)
    
    # 像切蛋糕一样：前 N 个给训练集，紧接着的 M 个给测试集，绝对不会重叠
    magpie_train_sampled = magpie_deduped[:num_train_needed]
    magpie_test_sampled = magpie_deduped[num_train_needed : num_train_needed + num_test_needed]
    
    print(f"✅ 为训练集抽取了 {len(magpie_train_sampled)} 条独立合成数据")
    print(f"✅ 为测试集抽取了 {len(magpie_test_sampled)} 条独立合成数据")
    
    # 4. 添加标签并合并
    print("\n🏷️ 添加真假标签 (真实=0, 合成=1)...")
    
    for item in oasst2_train:
        item['label'] = 0
        item['source'] = 'oasst2'
    for item in oasst2_test:
        item['label'] = 0
        item['source'] = 'oasst2'
        
    for item in magpie_train_sampled:
        item['label'] = 1
        item['source'] = 'magpie'
    for item in magpie_test_sampled:
        item['label'] = 1
        item['source'] = 'magpie'
        
    # 合并
    train_combined = oasst2_train + magpie_train_sampled
    test_combined = oasst2_test + magpie_test_sampled
    
    # 混合打乱 (确保分类器训练时正负样本交替出现)
    np.random.seed(seed)
    np.random.shuffle(train_combined)
    np.random.shuffle(test_combined)
    
    # 5. 保存数据
    print("\n💾 保存最终数据集...")
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, 'train.jsonl')
    test_file = os.path.join(output_dir, 'test.jsonl')
    
    save_jsonl(train_combined, train_file)
    save_jsonl(test_combined, test_file)
    
    print(f"✅ 训练集: {train_file} (总计 {len(train_combined)} 条)")
    print(f"✅ 测试集: {test_file} (总计 {len(test_combined)} 条)")
    
    # 6. 生成简单的统计信息
    stats = {
        "method": "random_sampling_with_deduplication",
        "random_seed": seed,
        "train_set": {
            "total": len(train_combined),
            "real_count": len(oasst2_train),
            "synthetic_count": len(magpie_train_sampled)
        },
        "test_set": {
            "total": len(test_combined),
            "real_count": len(oasst2_test),
            "synthetic_count": len(magpie_test_sampled)
        },
        "magpie_deduplication_stats": {
            "original_count": len(magpie_all),
            "deduped_count": len(magpie_deduped),
            "duplicates_removed": len(magpie_all) - len(magpie_deduped)
        }
    }
    
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*80)
    print("🎉 数据集随机抽取与合并完美收工！")
    print(f"📊 统计报告已保存至: {stats_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="纯随机抽取并合并 OASST2 和 Magpie 数据集 (包含去重与防泄露机制)"
    )
    
    parser.add_argument('--oasst2_train', type=str, default=DEFAULT_OASST2_TRAIN,
                        help=f'OASST2 训练集文件 (默认: {DEFAULT_OASST2_TRAIN})')
    parser.add_argument('--oasst2_test', type=str, default=DEFAULT_OASST2_TEST,
                        help=f'OASST2 测试集文件 (默认: {DEFAULT_OASST2_TEST})')
    parser.add_argument('--magpie_all', type=str, default=DEFAULT_MAGPIE_ALL,
                        help=f'Magpie 全部数据文件 (默认: {DEFAULT_MAGPIE_ALL})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'随机种子 (默认: {RANDOM_SEED})')
    
    args = parser.parse_args()
    
    sample_and_merge(
        oasst2_train_file=args.oasst2_train,
        oasst2_test_file=args.oasst2_test,
        magpie_all_file=args.magpie_all,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()