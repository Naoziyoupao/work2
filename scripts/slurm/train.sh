#!/bin/bash
#SBATCH -J train
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00

# ==================== 任务说明 ==================== #
# 实验003：OASST2合成数据生成
# 使用Qwen3-32B模型生成5000条合成数据
# Best-of-N采样 (N=64) + ArmoRM打分 + 长度对齐
# ================================================== #

echo "=========================================="
echo "开始时间: $(date)"
echo "节点: $SLURM_NODELIST"
echo "任务ID: $SLURM_JOB_ID"
echo "GPU数量: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# 激活conda环境
conda activate qwen3

# 检查环境
echo ""
echo "Python版本:"
python --version
echo ""
echo "PyTorch版本:"
python -c "import torch; print(torch.__version__)"
echo ""
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""


# 创建日志目录
mkdir -p logs


# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 运行合成数据生成脚本
echo "=========================================="
echo "开始生成合成数据..."
echo "=========================================="

/home/xzhang/anaconda3_new/envs/qwen3/bin/python /home/xzhang/工作2/scripts/train/train_classifier_new.py \
    --train_file /home/xzhang/工作2/data/processed/classification_random/train.jsonl \
    --val_file /home/xzhang/工作2/data/processed/classification_random/test.jsonl \
    --output_dir /home/xzhang/工作2/outputs/classification_random

