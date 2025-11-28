import json
import random
import os

# --- Configuration ---
input_file = "data/PopMC_0_difficulty_filtered.jsonl"

# Ratios (Must sum to 1.0)
train_ratio = 0.80
val_ratio = 0.10
test_ratio = 0.10

# --- Execution ---
print(f"Reading from {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# 1. Shuffle (Reproducible)
random.seed(42)
random.shuffle(lines)

# 2. Calculate Indices
total = len(lines)
train_end = int(total * train_ratio)
val_end = int(total * (train_ratio + val_ratio))

# 3. Slice
train_data = lines[:train_end]
val_data = lines[train_end:val_end]
test_data = lines[val_end:]

# 4. Generate output file paths
base_name = input_file.replace('.jsonl', '')
paths = {
    "train": f"{base_name}_train.jsonl",
    "val":   f"{base_name}_val.jsonl",
    "test":  f"{base_name}_test.jsonl"
}

with open(paths["train"], 'w', encoding='utf-8') as f:
    f.write('\n'.join(train_data) + '\n')

with open(paths["val"], 'w', encoding='utf-8') as f:
    f.write('\n'.join(val_data) + '\n')

with open(paths["test"], 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_data) + '\n')

print("Done! Files generated:")
print(f"  Train: {paths['train']} - {len(train_data)} questions")
print(f"  Val:   {paths['val']} - {len(val_data)} questions")
print(f"  Test:  {paths['test']} - {len(test_data)} questions")