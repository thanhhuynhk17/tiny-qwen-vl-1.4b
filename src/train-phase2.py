import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyQwenVL model - phase 2")

    # Add your arguments here
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_path", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    # add checkpoint path
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/best_tinyqwenvl_1.4B.pth", help="Path to the best model checkpoint")
    return parser.parse_args()

args = parse_args()

# Now you can use args.batch_size, args.learning_rate, etc.
print(f"Batch size: {args.batch_size}")
print(f"Learning rate: {args.learning_rate}")
print(f"Saving path: {args.save_path}")


SIGLIP_MODEL = "google/siglip-so400m-patch14-224"
QWEN_MODEL = "Qwen/Qwen2.5-0.5B"

from transformers import set_seed
set_seed(args.seed)  # Set random seed for reproducibility

# LOAD dataset
from datasets import load_dataset

dataset = load_dataset("lmms-lab/COCO-Caption", split="val")
drop_cols = ['question_id',
             'id',
             'license',
             'file_name',
             'coco_url',
             'height',
             'width',
             'date_captured']
dataset = dataset.remove_columns(drop_cols)

from transformers import AutoProcessor, AutoTokenizer

processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
processor.image_processor.size = {'height': 448, 'width': 448}
print("processor config:", processor.image_processor)

tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
# add special tokens extended
tokenizer.add_special_tokens({
    'pad_token': '<|endoftext|>',
    # Notes: Adding unknown special tokens is not supported
    # <|extra_0|> as begin image token
    # <|extra_1|> as end image token
    'additional_special_tokens': ['<|extra_0|>', '<|extra_1|>']
})
tokenizer.padding_side='right'

# Split dataset
from src.preprocess import collate_fn

split_dataset = dataset.train_test_split(test_size=0.2, seed=args.seed)
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
batch_size = args.batch_size
val_loader = DataLoader(split_dataset["test"], batch_size=batch_size, shuffle=False,\
                        collate_fn=lambda batch: collate_fn(batch,
                                                            processor,
                                                            tokenizer))
# free memory by del dataset, split_dataset
del dataset

## Show 1 sample 
sample = next(iter(val_loader))
for key in sample.keys():
    print(f'{key}\t:\t{sample[key].shape}')
print("Sample pixel_values:", sample["pixel_values"][0].shape)
print("Sample input_ids:", sample["input_ids"][0])
print("Sample attention_mask:", sample["attention_mask"][0])
print("Sample labels:", sample["labels"][0])

# LOAD model
from src.model import TinyQwenVL 
import torch

# Automatically select dtype based on GPU capability
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("Using bfloat16")
else:
    dtype = torch.float32
    print("Using float32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = TinyQwenVL(torch_dtype=dtype)
model.text_decoder.resize_token_embeddings(len(tokenizer))

# load phase1 checkpoint
state_dict = torch.load(args.ckpt_path, map_location=device)
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)
print("adapter._precomputed_rope: ", model.adapter._precomputed_rope)
if model.adapter._precomputed_rope == True:
    model.adapter._precomputed_rope = False

# model text decoder require gradients
for param in model.text_decoder.parameters():
    print("param.requires_grad:", param.requires_grad)
    break

model.to(device)

# get model stats
print("Model's stats:\n",model.get_model_stats())

# Init optimizer
from torch import nn
import torch.optim as optim
# import evaluate
# accuracy = evaluate.load("accuracy")

optimizer = optim.AdamW(model.parameters(), 
                        lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
# Note: CrossEntropyLoss expects the input to be of shape (N, C) 
# where N is the batch size and C is the number of classes.
# Training loop
from torch.nn.utils import clip_grad_norm_

num_epochs = args.num_epochs
best_val_loss = float('inf')
best_model_path = f"{args.save_path}/best_tinyqwenvl_1.4B-phase2.pth"

gradient_accumulation_steps = args.gradient_accumulation_steps  # ga x bz = real batch_size to take gradient backward

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    total_loss = 0

    train_loader = DataLoader(split_dataset["train"], batch_size=batch_size, shuffle=True,\
                            collate_fn=lambda batch: collate_fn(batch,
                                                                processor,
                                                                tokenizer))
    
    print(f"\n🚀 Epoch {epoch+1}/{num_epochs} Training...\n")
    # ─── TRAINING PHASE ─────────────────────────────────────────────────────
    for step, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    interpolate_pos_encoding=True,
                )

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous() # [B, total_seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous() # [B, total_seq_len-1]
        # Flatten the tokens
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)),
                         shift_labels.view(-1))
        # tracking raw loss
        raw_loss = loss.item()
        total_loss += raw_loss
        # Normalize
        loss = loss / gradient_accumulation_steps

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)

        # Update Optimizer
        if ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"🟢 Step {step+1}/{len(train_loader)} | Raw loss: {raw_loss:.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n📊 Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
    # ─── TRAINING PHASE ─────────────────────────────────────────────────────


    # ─── VALIDATION PHASE ─────────────────────────────────────────────────────
    model.eval()
    total_eval_loss = 0.0
    best_eval_loss = float('inf')

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            # val inputs for loss computation
            pixel_values     = batch["pixel_values"].to(device)
            input_ids        = batch["input_ids"].to(device)
            attention_mask   = batch["attention_mask"].to(device)
            labels           = batch["labels"].to(device)            # (B, seq_len)

            # Compute validation loss (same as training)
            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                interpolate_pos_encoding=True,
            )

            # Loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_eval_loss += loss.item()

    avg_loss = total_eval_loss / len(val_loader)
    print(f"\n📊 Validation Results | Loss: {avg_loss:.4f}")

    # Save best model by best eval loss
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), f"{best_model_path}")
        print(f"✅ New best checkpoint saved {best_model_path} {best_val_loss:.4f}")