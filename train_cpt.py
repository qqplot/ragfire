#!/usr/bin/env python3
# type: ignore
"""
OpenSloth - Configurable training script with wandb logging.
MODIFIED FOR CPT (Continued Pretraining) + DDP (torchrun)
"""

import os
import argparse
import glob
import torch
from typing import Any, Tuple, Optional
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

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
    parser = argparse.ArgumentParser(description="Configurable OpenSloth training (CPT Mod, DDP-ready)")
    # 1.7B & 16K
    parser.add_argument("--model_name", default="unsloth/Qwen3-1.7B", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=16384, help="Max sequence length")

    # steps/epochs
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=12719, help="Max steps (overrides epochs)")

    # CPT hparams
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--per_device_batch_size", type=int, default=4, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")

    # misc
    parser.add_argument("--train_samples", type=int, default=-1, help="(ignored for CPT)")
    parser.add_argument("--test_samples", type=int, default=0, help="(ignored for CPT)")
    parser.add_argument("--experiment_name", default="opensloth_cpt", help="Experiment name")

    # dataset prepare
    parser.add_argument("--skip_prepare", action="store_true", help="Use pre-processed dataset")
    parser.add_argument("--prepared_dataset_path", type=str, help="Path to pre-processed dataset directory")

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
        device_map=None,  # DDP에서는 None 권장 (각 프로세스가 자신의 GPU 담당)
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=256,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
            "lm_head","embed_tokens",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
        temporary_location=f"_unsloth_temp_buffers/rank_{local_rank}",
    )
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# -----------------------------
# Data
# -----------------------------
def prepare_datasets(args, tokenizer: Any) -> Tuple[Dataset, Optional[Dataset]]:
    """Load and format CPT dataset (.jsonl). Uses streaming + sharding for DDP."""
    world_size, local_rank, rank, is_main, device = get_dist_info()

    if args.skip_prepare:
        if not args.prepared_dataset_path:
            raise ValueError("--prepared_dataset_path must be specified when using --skip_prepare")

        if is_main:
            print(f"📊 Loading pre-processed dataset from: {args.prepared_dataset_path}")

        from pathlib import Path
        dataset_path = Path(args.prepared_dataset_path)
        train_path = dataset_path / "train"
        if not train_path.exists():
            raise FileNotFoundError(f"Training dataset not found at: {train_path}")

        train_dataset = load_from_disk(str(train_path))
        if is_main:
            print(f"✅ Loaded {len(train_dataset)} pre-processed training examples")
            print("ℹ️ CPT mode: No validation dataset loaded.")
        return train_dataset, None

    # streaming load
    if is_main:
        print("📊 Loading CPT dataset from: /home/qqplot/ragfire/data/clean/")
    jsonl_files = glob.glob("/home/qqplot/ragfire/data/clean/**/*.jsonl", recursive=True)
    if not jsonl_files:
        raise ValueError("No .jsonl files found in /home/qqplot/ragfire/data/clean/")

    dataset = load_dataset("json", data_files=jsonl_files, split="train", streaming=True)

    # IMPORTANT: shuffle + shard for DDP
    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    dataset = dataset.shard(num_shards=world_size, index=rank)

    # get original columns (consume one example then re-chain it)
    first = next(iter(dataset))
    original_columns = list(first.keys())

    # Recreate an iterator that includes 'first' back
    def prepend_first(gen, first_item):
        yield first_item
        for x in gen:
            yield x
    dataset = prepend_first(dataset, first)  # generator

    # tokenization
    def tokenize_function(examples):
        texts_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(texts_with_eos, return_attention_mask=False)

    # group/paking
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
        result = {
            k: [t[i:i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_dataset = load_dataset(
        "json",
        data_files=jsonl_files,
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000).shard(num_shards=world_size, index=rank).map(
        tokenize_function, batched=True, remove_columns=original_columns
    )

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)

    if is_main:
        print("✅ CPT Dataset pipeline ready (streaming + sharded).")
    return lm_dataset, None


# -----------------------------
# Print config
# -----------------------------
def print_configuration(args) -> None:
    world_size, _, _, _, _ = get_dist_info()
    effective_batch = args.per_device_batch_size * args.grad_accum * world_size
    print("📊 CPT Training configuration:")
    print(f"   - Model: {args.model_name}")
    print(f"   - Max sequence length: {args.max_seq_length}")
    print(f"   - Epochs: {args.num_epochs} (if max_steps not set)")
    print(f"   - Max steps: {args.max_steps} (overrides epochs)")
    print(f"   - World size: {world_size} GPU(s)")
    print(f"   - Per device batch size: {args.per_device_batch_size}")
    print(f"   - Gradient accumulation: {args.grad_accum}")
    print(f"   - Effective batch size: {effective_batch}")
    print(f"   - Learning rate: {args.learning_rate}")


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

    training_args = UnslothTrainingArguments(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 999,
        embedding_learning_rate=5e-6,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        weight_decay=0.00,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),        
        gradient_checkpointing=True,
        optim="adamw_8bit",
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        report_to="none",
        dataloader_drop_last=True,
        torch_compile=False,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing = False, # Can make training 5x faster for short sequences.        
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
        print(f"🚀 Initializing training with DDP on {world_size} GPU(s)")

    print_configuration(args)

    if 'model' not in dir():
        model, tokenizer = init_model(args)

    train_dataset, val_dataset = prepare_datasets(args, tokenizer)
    trainer = build_trainer(args, model, tokenizer, train_dataset, val_dataset)

    # GPU stats per-rank
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(local_rank)
        start_reserved = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        print(f"[rank {rank}] GPU={gpu_props.name} | total={max_memory} GB | reserved={start_reserved} GB")

    if is_main:
        print("🎯 Starting CPT training...")

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
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            print(f"✅ Merged model saved to {merged_model_path}")
        except Exception as e:
            print(f"🔥 Failed to merge model: {e}")
