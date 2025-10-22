#!/usr/bin/env python3
# type: ignore
"""
OpenSloth - Configurable training script with wandb logging.
MODIFIED FOR SFT (Supervised Fine-Tuning) + DDP (torchrun)
"""

import os
import argparse
import torch
from typing import Any, Tuple, Optional
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
# from unsloth import UnslothTrainer, UnslothTrainingArguments
from trl import SFTConfig, SFTTrainer

from datasets import Dataset, load_dataset, load_from_disk


# -----------------------------
# Dist helpers
# -----------------------------
def get_dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return world_size, local_rank, rank, is_main, device


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Configurable OpenSloth training (SFT Mod, DDP-ready)")
    # 1.7B & 16K
    parser.add_argument("--model_name", default="unsloth/Qwen3-1.7B", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Max sequence length")

    # steps/epochs
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs)")

    # SFT hparams
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (SFT default)")
    parser.add_argument("--per_device_batch_size", type=int, default=2, help="Per device batch size (SFT default)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps") # BS=2, GA=4 -> 8

    # misc
    parser.add_argument("--experiment_name", default="opensloth_sft_rag", help="Experiment name")

    # dataset
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="rag_sft_dataset.jsonl", 
        help="Path to the SFT dataset (.jsonl file)"
    )
    parser.add_argument(
        "--validation_split_percentage", 
        type=float, 
        default=0.05, 
        help="Percentage of data to use for validation (e.g., 0.05 for 5%)"
    )

    return parser.parse_args()


# -----------------------------
# Model
# -----------------------------
def init_model(args) -> Tuple[FastLanguageModel, Any]:
    world_size, local_rank, rank, is_main, device = get_dist_info()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        device_map=None,  # DDP에서는 None 권장
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
            # "lm_head","embed_tokens", # SFT에서는 이 부분들을 제외하는 것이 일반적
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
        temporary_location=f"_unsloth_sft_temp_buffers/rank_{local_rank}",
    )
    model.to(device)

    # Tokenizer pad token 설정은 SFT에서도 중요합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# -----------------------------
# Data
# -----------------------------
def prepare_datasets(args) -> Tuple[Dataset, Optional[Dataset]]:
    """Load SFT dataset (.jsonl) and split for DDP."""
    world_size, local_rank, rank, is_main, device = get_dist_info()

    if is_main:
        print(f"📊 Loading SFT dataset from: {args.dataset_path}")

    # 1. JSONL 파일 로드
    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    except FileNotFoundError:
        print(f"오류: SFT 데이터셋 파일을 찾을 수 없습니다: {args.dataset_path}")
        raise

    # 2. DDP를 위한 샤딩 (각 GPU가 데이터의 일부만 갖도록)
    dataset = dataset.shard(num_shards=world_size, index=rank)
    
    # 3. 훈련/검증 데이터셋 분리
    val_dataset = None
    if args.validation_split_percentage > 0:
        if is_main:
            print(f"splitting dataset ({args.validation_split_percentage * 100}% validation)")
        split_data = dataset.train_test_split(
            test_size=args.validation_split_percentage, 
            seed=42
        )
        train_dataset = split_data["train"]
        val_dataset = split_data["test"]
    else:
        train_dataset = dataset
        if is_main:
            print("ℹ️ Validation split 0, using full dataset for training.")

    if is_main:
        print(f"✅ SFT Dataset loaded.")
        print(f"   - Total examples (rank {rank}): {len(dataset)}")
        print(f"   - Train examples (rank {rank}): {len(train_dataset)}")
        if val_dataset:
            print(f"   - Valid examples (rank {rank}): {len(val_dataset)}")
            
    # SFT에서는 UnslothTrainer가 내부적으로 토크나이징과 템플릿 적용을 처리합니다.
    # 'messages' 컬럼만 있으면 됩니다.
    
    return train_dataset, val_dataset


# -----------------------------
# Print config
# -----------------------------
def print_configuration(args) -> None:
    world_size, _, _, _, _ = get_dist_info()
    effective_batch = args.per_device_batch_size * args.grad_accum * world_size
    print("📊 SFT Training configuration:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Max sequence length: {args.max_seq_length}")
    print(f"   - Epochs: {args.num_epochs}")
    print(f"   - Max steps: {args.max_steps}")
    print(f"   - World size: {world_size} GPU(s)")
    print(f"   - Per device batch size: {args.per_device_batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Validation split: {args.validation_split_percentage * 100}%")
    print(f"   - Dataset: {args.dataset_path}")


# -----------------------------
# Trainer
# -----------------------------
def build_trainer(
    args,
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
) -> UnslothTrainer:
    world_size, local_rank, rank, is_main, device = get_dist_info()
    run_name = f"{args.experiment_name}_ddp{world_size}"

    training_args = SFTConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 999,
        # embedding_learning_rate 제거
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset is not None else "no",
        eval_steps=200 if val_dataset is not None else None,
        weight_decay=0.00,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),        
        gradient_checkpointing=True,
        optim="adamw_8bit",
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        report_to="none", # "wandb"로 변경하여 wandb 로깅 가능
        dataloader_drop_last=True,
        torch_compile=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # dataset_text_field="text", # SFT에서는 제거 (messages 컬럼 자동 감지)
        max_seq_length=args.max_seq_length,
        # packing = False, # SFT에서는 제거 (ChatML 템플릿이 적용됨)
        args=training_args,
    )
    return trainer


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    world_size, local_rank, rank, is_main, device = get_dist_info()

    if is_main:
        print(f"🚀 Initializing SFT training with DDP on {world_size} GPU(s)")

    print_configuration(args)

    if 'model' not in dir():
        model, tokenizer = init_model(args)

    # SFT 데이터 준비 (tokenizer 필요 없음)
    train_dataset, val_dataset = prepare_datasets(args)
    
    trainer = build_trainer(args, model, tokenizer, train_dataset, val_dataset)

    # GPU stats per-rank
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(local_rank)
        start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        print(f"[rank {rank}] GPU={gpu_props.name} | total={max_memory} GB | reserved={start_reserved} GB")

    if is_main:
        print("🎯 Starting SFT training...")

    trainer.train()

    if is_main:
        print("✅ Training completed!")
        print("Saving final adapter...")
        final_model_path = f"outputs/{trainer.args.run_name}/final_adapter"
        os.makedirs(final_model_path, exist_ok=True)
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✅ Final QLoRA adapter saved to {final_model_path}")

        print("Merging adapter...")
        try:
            merged_model_path = f"outputs/{trainer.args.run_name}/final_merged"
            os.makedirs(merged_model_path, exist_ok=True)
            # DDP 훈련 후에는 model.merge_and_unload() 전에 trainer.model을 사용하거나,
            # 모델을 다시 로드하는 것이 더 안전할 수 있습니다.
            # 여기서는 원본 스크립트의 로직을 따릅니다.
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            print(f"✅ Merged model saved to {merged_model_path}")
        except Exception as e:
            print(f"🔥 Failed to merge model: {e}")