# LLM 研究项目 - 分类器训练精简版

这是一个大语言模型（LLM）相关的精简研究项目，专注于训练基于 LLaMA 的二分类器，用于区分“真实世界答案”与“合成答案”。

## 📚 快速导航

### 📖 核心文档

| 文档 | 描述 | 链接 |
|-----|------|-----|
| 🎯 项目说明 | 项目目标、保留范围、迁移说明 | [docs/00-项目说明.md](docs/00-项目说明.md) |

## 🚀 快速开始

### 1. 激活环境

```bash
conda activate qwen3
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. （可选）处理 OASST2 数据

```bash
python scripts/data/process_oasst2.py
python scripts/data/prepare_classification_data.py
python scripts/data/prepare_classification_data_response_only.py
```

### 3.5 （可选）下载和处理 Magpie 数据集（作为合成数据源）

```bash
# 下载 Magpie 数据集
python scripts/data/download_magpie.py

# 处理 Magpie 数据集
python scripts/data/process_magpie.py

# 使用 Magpie 作为合成数据准备分类数据
python scripts/data/prepare_classification_data.py \
    --real_train data/processed/oasst2/first_turn_train_5k.jsonl \
    --real_val data/processed/oasst2/first_turn_val_5k.jsonl \
    --synthetic_train data/processed/magpie/synthetic_train.jsonl \
    --synthetic_val data/processed/magpie/synthetic_val.jsonl \
    --output_dir data/processed/magpie_classification
```

### 4. 训练分类器

```bash
python /home/xzhang/工作2/scripts/train/train_classifier_new.py \
    --train_file data/processed/classification_random/train.jsonl \
    --val_file data/processed/classification_random/test.jsonl \
    --cache_dir ./cache \
    --output_dir ./outputs/classification_random
```

## 📂 项目结构

```
/home/xzhang/工作2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── train/
│   │   ├── train_classifier_new.py
│   │   └── train_classifier.py
│   └── data/
│       ├── process_oasst2.py
│       ├── download_magpie.py
│       ├── process_magpie.py
│       ├── prepare_classification_data.py
│       └── prepare_classification_data_response_only.py
│
├── data/
│   ├── raw/
│   │   ├── oasst2/
│   │   └── magpie/
│   └── processed/
│       ├── oasst2/
│       └── magpie/
│
├── docs/
│   └── 00-项目说明.md
└── configs/
```

## 💡 使用建议

1. 本仓库仅保留“训练分类器”主流程。
2. 已移除分析类/困惑度/PCA/多实验对比等内容。
3. OASST2 数据目录结构已按原工程保留到新路径。

## 📊 数据集

本项目支持以下数据集：

### 真实数据集
- **OASST2** - 真实人类对话数据
  - `data/raw/oasst2/`
  - `data/processed/oasst2/`

### 合成数据集
- **Magpie-Pro-300K-Filtered** - 高质量合成对话数据
  - `data/raw/magpie/`
  - `data/processed/magpie/`

## 🎯 当前任务进度

### Phase 1：迁移与精简 ✅
- [x] 新建独立目录 `/home/xzhang/工作2`
- [x] 保留 OASST2 数据结构
- [x] 仅保留分类器训练脚本
- [x] 重写 README（保留原风格）
- [x] 初始化 Git 仓库

### Phase 2：集成 Magpie 数据集 ✅
- [x] 创建 Magpie 数据下载脚本
- [x] 创建 Magpie 数据处理脚本
- [x] 支持 Magpie 作为合成数据源
- [x] 添加数据隔离配置（.clineignore）

## 📄 许可证

本项目仅用于学术研究目的。

---

**最后更新**: 2026-03-08  
**文档版本**: v1.0（分类器训练精简迁移版）