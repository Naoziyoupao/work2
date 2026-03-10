import os
import json
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

# --- 常量定义 --- #
dataset_path = "data/raw/oasst2"
output_dir = "data/processed/oasst2"

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

def path_to_multi_turn(path, conversation_id):
    """
    将对话路径转换为多轮对话格式
    
    Args:
        path: 消息路径
        conversation_id: 对话ID
    
    Returns:
        多轮对话字典，如果路径无效则返回None
    """
    if len(path) < 2:  # 至少需要一轮完整对话（user + assistant）
        return None
    
    messages = []
    for msg in path:
        role = msg.get('role')
        content = msg.get('text', '').strip()
        
        if not content:
            continue
        
        # 映射角色名称
        if role == 'prompter':
            role = 'user'
        elif role == 'assistant':
            role = 'assistant'
        else:
            continue
        
        messages.append({
            "role": role,
            "content": content
        })
    
    # 确保至少有一轮完整对话
    if len(messages) < 2:
        return None
    
    return {
        "conversation_id": conversation_id,
        "messages": messages
    }

def path_to_single_turns(path, conversation_id):
    """
    将对话路径转换为多个单轮问答对
    
    Args:
        path: 消息路径
        conversation_id: 对话ID
    
    Returns:
        单轮问答列表
    """
    single_turns = []
    
    for i in range(len(path) - 1):
        if path[i].get('role') == 'prompter' and path[i+1].get('role') == 'assistant':
            question = path[i].get('text', '').strip()
            answer = path[i+1].get('text', '').strip()
            
            if question and answer:
                single_turns.append({
                    "conversation_id": f"{conversation_id}_turn{len(single_turns)}",
                    "text": f"问题: {question}\n答案: {answer}",
                    "instruction": question,
                    "response": answer
                })
    
    return single_turns

def process_oasst2():
    """处理OASST2数据集"""
    print(f"🚀 开始加载数据集: {dataset_path}...")
    
    # 加载数据集
    try:
        dataset = load_from_disk(dataset_path)
        # OASST2通常只有train split
        if 'train' in dataset:
            messages = dataset['train']
        else:
            messages = dataset
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        print("提示：请先运行 download_data.py 下载数据集")
        return
    
    print(f"📊 数据集包含 {len(messages)} 条消息")
    
    # 1. 筛选英语消息
    print("🔍 正在筛选英语消息...")
    en_messages = []
    for msg in tqdm(messages, desc="筛选英语"):
        if msg.get('lang') == 'en':
            en_messages.append(msg)
    
    print(f"✅ 筛选出 {len(en_messages)} 条英语消息")
    
    # 2. 构建对话树
    print("🌳 正在构建对话树...")
    msg_dict, root_ids, children_map = build_conversation_trees(en_messages)
    print(f"📌 发现 {len(root_ids)} 个对话树根节点")
    
    # 3. 提取最优路径
    print("🎯 正在提取评分最高的对话路径...")
    multi_turn_conversations = []
    single_turn_pairs = []
    
    for root_id in tqdm(root_ids, desc="处理对话树"):
        # 获取最优路径
        path = get_best_path(root_id, msg_dict, children_map)
        
        # 转换为多轮对话格式
        multi_turn = path_to_multi_turn(path, root_id)
        if multi_turn:
            multi_turn_conversations.append(multi_turn)
        
        # 转换为单轮问答格式
        single_turns = path_to_single_turns(path, root_id)
        single_turn_pairs.extend(single_turns)
    
    print(f"✅ 提取出 {len(multi_turn_conversations)} 个多轮对话")
    print(f"✅ 提取出 {len(single_turn_pairs)} 个单轮问答对")
    
    # 4. 数据切分
    print("⚖️ 正在进行训练集/验证集切分 (9:1)...")
    
    # 切分多轮对话
    multi_train, multi_val = train_test_split(
        multi_turn_conversations, 
        test_size=0.1, 
        random_state=42
    )
    
    # 切分单轮问答
    single_train, single_val = train_test_split(
        single_turn_pairs,
        test_size=0.1,
        random_state=42
    )
    
    # 5. 保存数据
    os.makedirs(output_dir, exist_ok=True)
    
    print("💾 正在保存数据...")
    
    # 保存多轮对话
    multi_train_file = os.path.join(output_dir, "multi_turn_train.jsonl")
    multi_val_file = os.path.join(output_dir, "multi_turn_val.jsonl")
    
    with open(multi_train_file, 'w', encoding='utf-8') as f:
        for item in multi_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(multi_val_file, 'w', encoding='utf-8') as f:
        for item in multi_val:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存单轮问答
    single_train_file = os.path.join(output_dir, "single_turn_train.jsonl")
    single_val_file = os.path.join(output_dir, "single_turn_val.jsonl")
    
    with open(single_train_file, 'w', encoding='utf-8') as f:
        for item in single_train:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(single_val_file, 'w', encoding='utf-8') as f:
        for item in single_val:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 6. 输出统计信息
    print("\n" + "="*60)
    print("🎉 数据处理完成！")
    print("="*60)
    print("\n📊 统计信息:")
    print(f"  多轮对话:")
    print(f"    训练集: {len(multi_train)} 条")
    print(f"    验证集: {len(multi_val)} 条")
    
    # 计算平均对话轮次
    avg_turns = sum(len(c['messages']) for c in multi_turn_conversations) / len(multi_turn_conversations) / 2
    print(f"    平均对话轮次: {avg_turns:.2f}")
    
    print(f"\n  单轮问答:")
    print(f"    训练集: {len(single_train)} 条")
    print(f"    验证集: {len(single_val)} 条")
    
    print(f"\n📁 输出文件:")
    print(f"  {multi_train_file}")
    print(f"  {multi_val_file}")
    print(f"  {single_train_file}")
    print(f"  {single_val_file}")
    print("="*60)

if __name__ == "__main__":
    process_oasst2()