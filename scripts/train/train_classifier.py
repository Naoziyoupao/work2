import os
import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 配置区域 --- #
MODEL_ID = "/home/share/models/Llama-3.2-1B" 
TRAIN_FILE = "data/processed_openthoughts/val.jsonl"
VAL_FILE = "data/processed_openthoughts/val.jsonl"
OUTPUT_DIR = "./llama_classifier_lora_results"
PLOT_FILE = "training_metrics.png"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class LossPlottingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)

def train():
    # 1. 加载 Tokenizer
    print(f"正在加载 Tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 修复问题2: 设置左填充 (对于Decoder-only模型如Llama非常重要)
    tokenizer.padding_side = "left"

    # 2. 加载数据集并预加载到内存
    print("正在加载数据集到内存...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

    # 3. 数据预处理 (Tokenization)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    print("正在进行 Tokenization (预处理并保留在内存)...")
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["text"], # 移除原始文本以节省内存
        keep_in_memory=True # 显式保留在内存
    )

    # 4. 加载模型
    print(f"正在加载模型: {MODEL_ID}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=2,
        dtype=torch.bfloat16,  # 修复问题: 使用dtype替代已弃用的torch_dtype
        device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. LoRA 配置
    print("正在配置 LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 训练参数设置 - 修复所有问题
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        per_device_train_batch_size=4,  # 修复问题4: 减小批次大小
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 修复问题4: 添加梯度累积
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated for newer transformers versions,  # 修复问题5: 添加评估策略
        save_strategy="epoch",        # 修复问题5: 保存策略必须与评估策略匹配
        load_best_model_at_end=True,  # 修复问题5: 加载验证集上表现最好的模型
        metric_for_best_model="accuracy",
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        logging_dir=f"{OUTPUT_DIR}/logs",
        overwrite_output_dir=False,
    )

    # 7. 初始化 Trainer
    loss_callback = LossPlottingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[loss_callback]
    )

    # 8. 开始训练 (修复问题3: 使用Trainer内置的断点续传机制)
    print("🚀 开始训练...")
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"检测到历史 checkpoint: {last_checkpoint}，准备恢复训练...")
        else:
            print("未检测到历史 checkpoint，从头开始训练...")

    # 10. 绘制 Loss 曲线
    if loss_callback.losses:
        print(f"正在生成训练曲线图: {PLOT_FILE}")
        plt.figure(figsize=(10, 5))
        plt.plot(loss_callback.steps, loss_callback.losses, label="Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.savefig(PLOT_FILE)
        plt.close()

    # 11. 保存最终模型 (修复问题1: 确保保存分类头)
    print(f"✅ 训练完成，模型已保存至 {OUTPUT_DIR}")
    
    # 保存完整模型状态，包括分类头
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    # 同时保存tokenizer
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    
    # 也保存LoRA适配器参数（可选）
    trainer.save_model(os.path.join(OUTPUT_DIR, "lora_adapter"))

if __name__ == "__main__":
    train()