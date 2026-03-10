#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 OASST2 数据集 - 多轮对话拆分版本

将 OASST2 多轮对话中的每一轮都拆分为独立的问答对，作为真实数据

功能：
1. 读取原始 OASST2 数据
2. 筛选英语消息
3. 构建对话树并提取最优路径
4. 将每一轮对话都拆分为独立样本
5. 计算单词数等统计信息
6. 划分训练集/测试集（9:1）
"""

import os
import json
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import numpy as np

# 常量定义
DATASET_PATH = "data/raw/oasst2"
OUTPUT_DIR = "data/processed/oasst2"
RANDOM_SEED = 42
VAL_RATIO = 0.1


def build_conversation_trees(messages):
    """
    构建对话树结构
    
    Args:
        messages: 消息列表
    
    Returns:
        msg_dict: 消息ID到消息对象的映射
        root_ids: 根节点ID列表
        children_map: 父节点ID到子节点列表的映射
    """
    msg_dict = {}
    root_ids = []
    children_map = defaultdict(list)
    
    # 构建消息字典和树结构
    for msg in messages:
        msg_id = msg['message_id']
        msg_dict[msg_id] = msg
        
        parent_id = msg.get('parent_id')
        if parent_id is None:
            root_ids.append(msg_id)
        else:
            children_map[parent_id].append(msg_id)
    
    return msg_dict, root_ids, children_map


def get_best_path(msg_id, msg_dict, children_map):
    """
    从指定节点开始，递归获取评分最高的路径
    
    Args:
        msg_id: 当前消息ID
        msg_dict: 消息字典
        children_map: 子节点映射
    
    Returns:
        path: 从当前节点到叶子节点的最优路径（消息列表）
    """
    current_msg = msg_dict[msg_id]
    path = [current_msg]
    
    # 获取子节点
    children = children_map.get(msg_id, [])
    
    if not children:
        # 叶子节点，返回当前路径
        return path
    
    # 选择rank最小的子节点（rank越小评分越高）
    best_child = None
    best_rank = float('inf')
    
    for child_id in children:
        child_msg = msg_dict[child_id]
        child_rank = child_msg.get('rank')
        
        # 处理 rank 为 None 的情况
        if child_rank is None:
            child_rank = float('inf')
        
        if child_rank < best_rank:
            best_rank = child_rank
            best_child = child_id
    
    # 递归获取最优子路径
    if best_child is not None:
        child_path = get_best_path(best_child, msg_dict, children_map)
        path.extend(child_path)
    
    return path


def extract_all_turn_pairs(path, conversation_id):
    """
    从对话路径中提取所有轮次的问答对
    每一轮 (user -> assistant) 作为独立样本
    
    Args:
        path: 对话路径（消息列表）
        conversation_id: 对话ID
    
    Returns:
        pairs: 问答对列表
    """
    pairs = []
    
    # 直接遍历相邻的消息对
    for i in range(len(path) - 1):
        # 检查是否为 prompter -> assistant 的配对
        if path[i].get('role') == 'prompter' and path[i+1].get('role') == 'assistant':
            instruction = path[i].get('text', '').strip()
            response = path[i+1].get('text', '').strip()
            
            # 跳过空内容
            if not instruction or not response:
                continue
            
            # 计算单词数（使用空格分割）
            instruction_word_count = len(instruction.split())
            response_word_count = len(response.split())
            
            turn_index = len(pairs)
            
            pairs.append({
                'conversation_id': f"{conversation_id}_turn{turn_index}",
                'original_conversation_id': conversation_id,
                'instruction': instruction,
                'response': response,
                'instruction_word_count': instruction_word_count,
                'response_word_count': response_word_count,
                'turn_index': turn_index,
                'is_first_turn': turn_index == 0,
                'message_ids': {
                    'user': path[i].get('message_id'),
                    'assistant': path[i+1].get('message_id')
                }
            })
    
    return pairs


def process_oasst2_multiturn():
    """处理 OASST2 数据集 - 多轮对话拆分版本"""
    
    print("="*80)
    print("🚀 开始处理 OASST2 数据集（多轮对话拆分）")
    print("="*80)
    print(f"📂 数据路径: {DATASET_PATH}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🎲 随机种子: {RANDOM_SEED}")
    print(f"⚖️  测试集比例: {VAL_RATIO}")
    print("="*80)
    
    # 1. 加载数据集
    print(f"\n📥 加载 OASST2 数据集...")
    try:
        dataset = load_from_disk(DATASET_PATH)
        # OASST2 通常只有 train split
        if 'train' in dataset:
            messages = dataset['train']
        else:
            messages = dataset
        print(f"✅ 加载完成，共 {len(messages)} 条消息")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("💡 提示：请先运行 python scripts/data/download_magpie.py 下载数据集")
        return
    
    # 2. 筛选英语消息
    print(f"\n🔍 筛选英语消息...")
    en_messages = []
    for msg in tqdm(messages, desc="筛选英语"):
        if msg.get('lang') == 'en':
            en_messages.append(msg)
    
    print(f"✅ 筛选出 {len(en_messages)} 条英语消息")
    
    # 3. 构建对话树
    print(f"\n🌳 构建对话树...")
    msg_dict, root_ids, children_map = build_conversation_trees(en_messages)
    print(f"✅ 发现 {len(root_ids)} 个对话树根节点")
    
    # 4. 提取所有轮次的问答对
    print(f"\n🎯 提取所有轮次的问答对...")
    all_pairs = []
    conversations_processed = 0
    
    for root_id in tqdm(root_ids, desc="处理对话树"):
        # 获取最优路径
        path = get_best_path(root_id, msg_dict, children_map)
        
        # 提取所有轮次
        pairs = extract_all_turn_pairs(path, root_id)
        
        if pairs:
            all_pairs.extend(pairs)
            conversations_processed += 1
    
    print(f"✅ 从 {conversations_processed} 个对话中提取出 {len(all_pairs)} 个问答对")
    
    # 5. 统计信息
    print(f"\n📊 计算统计信息...")
    
    first_turn_pairs = [p for p in all_pairs if p['is_first_turn']]
    non_first_turn_pairs = [p for p in all_pairs if not p['is_first_turn']]
    
    instruction_word_counts = [p['instruction_word_count'] for p in all_pairs]
    response_word_counts = [p['response_word_count'] for p in all_pairs]
    
    stats = {
        "total_pairs": len(all_pairs),
        "first_turn_pairs": len(first_turn_pairs),
        "non_first_turn_pairs": len(non_first_turn_pairs),
        "conversations_processed": conversations_processed,
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
    
    print(f"  总问答对: {stats['total_pairs']}")
    print(f"  第一轮: {stats['first_turn_pairs']}")
    print(f"  后续轮次: {stats['non_first_turn_pairs']}")
    print(f"  指令单词数: {stats['instruction_word_count']['mean']:.1f} ± {stats['instruction_word_count']['std']:.1f}")
    print(f"  响应单词数: {stats['response_word_count']['mean']:.1f} ± {stats['response_word_count']['std']:.1f}")
    
    # 6. 划分训练集/测试集
    print(f"\n⚖️  划分训练集/测试集 ({(1-VAL_RATIO)*100:.0f}:{VAL_RATIO*100:.0f})...")
    
    train_pairs, test_pairs = train_test_split(
        all_pairs,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED
    )
    
    print(f"✅ 训练集: {len(train_pairs)} 条")
    print(f"✅ 测试集: {len(test_pairs)} 条")
    
    # 更新统计信息
    stats['train'] = {
        "count": len(train_pairs),
        "response_word_count": {
            "mean": float(np.mean([p['response_word_count'] for p in train_pairs])),
            "median": float(np.median([p['response_word_count'] for p in train_pairs])),
            "std": float(np.std([p['response_word_count'] for p in train_pairs]))
        }
    }
    
    stats['test'] = {
        "count": len(test_pairs),
        "response_word_count": {
            "mean": float(np.mean([p['response_word_count'] for p in test_pairs])),
            "median": float(np.median([p['response_word_count'] for p in test_pairs])),
            "std": float(np.std([p['response_word_count'] for p in test_pairs]))
        }
    }
    
    # 7. 保存数据
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n💾 保存数据...")
    
    # 保存全部数据
    all_file = os.path.join(OUTPUT_DIR, 'real_multiturn_all.jsonl')
    with open(all_file, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"✅ 全部数据: {all_file}")
    
    # 保存训练集
    train_file = os.path.join(OUTPUT_DIR, 'real_multiturn_train.jsonl')
    with open(train_file, 'w', encoding='utf-8') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"✅ 训练集: {train_file}")
    
    # 保存测试集
    test_file = os.path.join(OUTPUT_DIR, 'real_multiturn_test.jsonl')
    with open(test_file, 'w', encoding='utf-8') as f:
        for pair in test_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"✅ 测试集: {test_file}")
    
    # 保存统计信息
    stats_file = os.path.join(OUTPUT_DIR, 'real_multiturn_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✅ 统计信息: {stats_file}")
    
    # 8. 打印摘要
    print("\n" + "="*80)
    print("🎉 OASST2 多轮对话拆分完成！")
    print("="*80)
    print(f"📊 数据摘要:")
    print(f"  总问答对: {len(all_pairs)}")
    print(f"  训练集: {len(train_pairs)} ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  测试集: {len(test_pairs)} ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"\n📈 响应长度统计 (单词数):")
    print(f"  均值: {stats['response_word_count']['mean']:.1f}")
    print(f"  中位数: {stats['response_word_count']['median']:.1f}")
    print(f"  标准差: {stats['response_word_count']['std']:.1f}")
    print(f"  范围: [{stats['response_word_count']['min']}, {stats['response_word_count']['max']}]")
    print(f"\n📁 输出文件:")
    print(f"  {all_file}")
    print(f"  {train_file}")
    print(f"  {test_file}")
    print(f"  {stats_file}")
    print("="*80)


if __name__ == "__main__":
    process_oasst2_multiturn()
