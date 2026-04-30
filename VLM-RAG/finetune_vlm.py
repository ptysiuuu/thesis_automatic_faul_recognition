"""
finetune_vlm.py
===============
LoRA finetuning of Qwen2.5-VL-7B on SoccerNet-MVFoul.

Why LoRA and not full finetuning:
  - 7B model has ~7B parameters → ~28GB in fp32, ~14GB in bf16
  - Full finetuning needs ~3-4x model size in optimizer states → doesn't fit
  - LoRA adds ~20-50M trainable parameters (rank 16, alpha 32)
  - All pretrained weights stay frozen → can't overfit on 2,319 samples
  - Only the LoRA adapters learn the foul-specific decision mapping

LoRA targets: attention Q, K, V, O projections in the language model
  (NOT the vision encoder — keep it frozen to preserve visual features)

Dependencies:
  pip install transformers peft trl bitsandbytes accelerate
  pip install qwen-vl-utils

Usage:
  python finetune_vlm.py \
    --dataset_dir /net/tscratch/people/plgaszos/vlm_dataset \
    --output_dir  /net/tscratch/people/plgaszos/vlm_finetuned \
    --model_name  Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_rank   16 \
    --max_epochs  5 \
    --batch_size  1 \
    --grad_accum  8
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoulVLMDataset(Dataset):
    """
    Loads JSONL prepared by prepare_vlm_dataset.py.
    Each sample: images (list of paths) + conversations (user/assistant).
    """

    def __init__(self, jsonl_path: str, processor, max_samples: int = None):
        self.processor = processor
        self.samples = []

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        images = []
        for img_path in sample["images"]:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                # Fallback: blank image
                images.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))

        # Build message in Qwen2-VL format
        # User turn: contains <image> placeholders + text
        # Assistant turn: JSON answer
        user_content   = sample["conversations"][0]["content"]
        assistant_text = sample["conversations"][1]["content"]

        # Qwen2-VL expects content as list of dicts with type image/text
        content_list = []
        # Parse the user content: split on <image> tokens
        parts = user_content.split("<image>")
        img_idx = 0
        for i, part in enumerate(parts):
            if part.strip():
                content_list.append({"type": "text", "text": part})
            if i < len(parts) - 1:  # there's an image after this part
                if img_idx < len(images):
                    content_list.append({"type": "image", "image": images[img_idx]})
                    img_idx += 1

        messages = [
            {"role": "user",      "content": content_list},
            {"role": "assistant", "content": assistant_text},
        ]

        return messages, images


def collate_fn(batch, processor):
    """
    Custom collate for variable-length multi-image batches.
    Tokenizes and pads the batch.
    """
    from qwen_vl_utils import process_vision_info

    all_messages = [item[0] for item in batch]
    all_images   = [item[1] for item in batch]

    # Apply chat template — includes both user and assistant turns
    # We need to compute labels for the assistant turn only
    texts = [
        processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        for msgs in all_messages
    ]

    # Get image tensors
    image_inputs_list = []
    for msgs in all_messages:
        img_inputs, _ = process_vision_info(msgs)
        image_inputs_list.extend(img_inputs if img_inputs else [])

    # Tokenize
    inputs = processor(
        text=texts,
        images=image_inputs_list if image_inputs_list else None,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )

    # Build labels: mask everything except the assistant response
    # Find the assistant token position in each sequence
    labels = inputs["input_ids"].clone()

    # Get the assistant token id (Qwen2-VL uses <|im_start|>assistant)
    assistant_token = processor.tokenizer.encode(
        "<|im_start|>assistant", add_special_tokens=False
    )

    for i in range(labels.shape[0]):
        seq = labels[i].tolist()
        # Find last occurrence of assistant start token sequence
        start_pos = -1
        for j in range(len(seq) - len(assistant_token), -1, -1):
            if seq[j:j+len(assistant_token)] == assistant_token:
                start_pos = j + len(assistant_token)
                break

        if start_pos == -1:
            # Couldn't find assistant turn — mask everything
            labels[i] = -100
        else:
            # Mask everything before assistant response
            labels[i, :start_pos] = -100

    inputs["labels"] = labels
    return inputs


class FoulVLMTrainer(Trainer):
    """Custom trainer with multi-image collation."""

    def __init__(self, processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.processor),
            num_workers=4,
            pin_memory=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset or self.eval_dataset
        return DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, self.processor),
            num_workers=4,
        )


# ---------------------------------------------------------------------------
# Main finetuning function
# ---------------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(
        args.model_name, trust_remote_code=True
    )

    # Load in bf16 — fits on single A100 40GB for 7B model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )

    # ---------------------------------------------------------------------------
    # LoRA configuration
    # ---------------------------------------------------------------------------
    # Target ONLY the language model attention layers, not the vision encoder.
    # This is critical: vision encoder weights stay frozen to preserve the
    # visual features learned on billions of image-text pairs.
    # We only teach the LM head to MAP visual features → foul decisions.
    #
    # Qwen2-VL architecture:
    #   model.visual      → Vision encoder (FROZEN — do not target)
    #   model.model       → Language model (train LoRA here)
    #   model.lm_head     → Output head (optionally train)
    #
    # LoRA rank 16, alpha 32 → ~0.3% of parameters trainable
    # This is the right regime for 2,319 training samples.
    # ---------------------------------------------------------------------------

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,   # alpha = 2 * rank is standard
        lora_dropout=0.1,
        bias="none",
        # Target only LM attention layers — NOT vision encoder
        target_modules=[
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
        ],
        # Explicitly exclude vision encoder
        modules_to_save=None,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected output: ~0.3-0.5% trainable parameters

    # Freeze vision encoder explicitly (belt and suspenders)
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    # ---------------------------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------------------------
    train_dataset = FoulVLMDataset(
        jsonl_path=os.path.join(args.dataset_dir, "train.jsonl"),
        processor=processor,
        max_samples=args.max_train_samples,
    )
    eval_dataset = FoulVLMDataset(
        jsonl_path=os.path.join(args.dataset_dir, "valid.jsonl"),
        processor=processor,
        max_samples=args.max_eval_samples,
    )

    # ---------------------------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------------------------
    # Key choices:
    #   - max_epochs=5: more than enough for LoRA on 2K samples
    #   - warmup: 10% of steps to stabilize early training
    #   - cosine schedule: matches your MVNetwork training
    #   - eval_steps: evaluate every 100 steps during training
    #   - load_best_model_at_end: prevent overfitting
    # ---------------------------------------------------------------------------

    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps     = steps_per_epoch * args.max_epochs
    warmup_steps    = max(10, total_steps // 10)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        dataloader_num_workers=4,
        remove_unused_columns=False,   # critical for multi-image batches
        report_to="none",
        gradient_checkpointing=True,   # saves memory for 7B model
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = FoulVLMTrainer(
        processor=processor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting LoRA finetuning...")
    trainer.train()

    # Save LoRA adapters (NOT the full model — only ~100MB)
    adapter_path = str(output_dir / "lora_adapters")
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"LoRA adapters saved to {adapter_path}")
    print("To load: model = PeftModel.from_pretrained(base_model, adapter_path)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",       required=True)
    parser.add_argument("--output_dir",        required=True)
    parser.add_argument("--model_name",
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_rank",         type=int,   default=16)
    parser.add_argument("--max_epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",        type=int,   default=1)
    parser.add_argument("--grad_accum",        type=int,   default=8)
    parser.add_argument("--lr",                type=float, default=2e-4)
    parser.add_argument("--max_train_samples", type=int,   default=None)
    parser.add_argument("--max_eval_samples",  type=int,   default=None)
    args = parser.parse_args()
    main(args)
