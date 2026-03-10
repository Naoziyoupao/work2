#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 Magpie 数据集

从 HuggingFace 下载 Magpie-Align/Magpie-Pro-300K-Filtered 数据集
并保存到本地目录 data/raw/magpie/
"""

import os
import argparse
from datasets import load_dataset

# 默认配置
DEFAULT_DATASET_NAME = "Magpie-Align/Magpie-Pro-300K-Filtered"
DEFAULT_OUTPUT_DIR = "data/raw/magpie"
DEFAULT_CACHE_DIR = "./cache"


def download_magpie(
    dataset_name: str = DEFAULT_DATASET_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR
):
    """
    下载 Magpie 数据集
    
    Args:
        dataset_name: HuggingFace 数据集名称
        output_dir: 输出目录
        cache_dir: 缓存目录
    """
    print("="*80)
    print("🚀 开始下载 Magpie 数据集")
    print("="*80)
    print(f"📦 数据集: {dataset_name}")
    print(f"📁 输出目录: {output_dir}")
    print(f"💾 缓存目录: {cache_dir}")
    print("="*80)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 下载数据集
        print("\n⏬ 正在从 HuggingFace 下载数据集...")
        print("   这可能需要几分钟时间，请耐心等待...")
        
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print(f"✅ 数据集下载完成！")
        
        # 打印数据集信息
        print(f"\n📊 数据集信息:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} 条数据")
        
        # 保存到本地磁盘
        print(f"\n💾 正在保存数据集到: {output_dir}")
        dataset.save_to_disk(output_dir)
        
        print(f"✅ 数据集已保存到磁盘")
        
        # 显示数据样例
        print(f"\n📝 数据样例（第一条）:")
        if 'train' in dataset:
            sample = dataset['train'][0]
        else:
            first_split = list(dataset.keys())[0]
            sample = dataset[first_split][0]
        
        print("-"*80)
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
        print("-"*80)
        
        print("\n" + "="*80)
        print("🎉 Magpie 数据集下载完成！")
        print("="*80)
        print(f"📁 数据保存位置: {output_dir}")
        print(f"\n💡 下一步:")
        print(f"   运行 python scripts/data/process_magpie.py 来处理数据")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"\n💡 提示:")
        print(f"   1. 检查网络连接")
        print(f"   2. 确认 HuggingFace datasets 库已安装: pip install datasets")
        print(f"   3. 如需访问私有数据集，请先登录: huggingface-cli login")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="下载 Magpie 数据集"
    )
    
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f'HuggingFace 数据集名称 (默认: {DEFAULT_DATASET_NAME})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=DEFAULT_CACHE_DIR,
        help=f'缓存目录 (默认: {DEFAULT_CACHE_DIR})'
    )
    
    args = parser.parse_args()
    
    download_magpie(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
