"""
Filter PopMC dataset to remove questions in difficulty bin 0 (easiest).

This script:
1. Reads PopMC.jsonl
2. Bins questions by difficulty using s_pop (subject popularity)
3. Filters out questions in difficulty bin 0 (easiest)
4. Saves remaining questions (bins 1-4) to PopMC_0_difficulty_filtered.jsonl
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/PopMC.jsonl")
output_file = Path("data/PopMC_0_difficulty_filtered.jsonl")

# Read all questions from PopMC.jsonl
print(f"Reading questions from {input_file}...")
questions = []
with open(input_file, 'r') as f:
    for line in f:
        if line.strip():  # Skip empty lines
            questions.append(json.loads(line))

print(f"Loaded {len(questions)} questions")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(questions)

# Calculate difficulty from s_pop (same method as in analyze_logprobs.ipynb)
# Lower s_pop = more obscure = higher difficulty
df['difficulty'] = -np.log1p(df['s_pop'])

# Create quantile-based difficulty bins (5 bins: 0=easiest, 4=hardest)
df['difficulty_bin'] = pd.qcut(df['difficulty'], q=5, labels=False)

# Show bin distribution
print("\nDifficulty bin distribution:")
print(df['difficulty_bin'].value_counts().sort_index())

# Filter out questions in difficulty bin 0 (easiest)
df_filtered = df[df['difficulty_bin'] != 0].copy()

print(f"\nFiltered out {len(df) - len(df_filtered)} questions from bin 0")
print(f"Remaining questions: {len(df_filtered)} (bins 1-4)")

# Remove the temporary columns before saving
df_filtered = df_filtered.drop(columns=['difficulty', 'difficulty_bin'])

# Write filtered questions to output file
print(f"\nWriting filtered questions to {output_file}...")
with open(output_file, 'w') as f:
    for _, row in df_filtered.iterrows():
        # Convert row back to dict and write as JSON line
        question_dict = row.to_dict()
        f.write(json.dumps(question_dict) + '\n')

print(f"Successfully wrote {len(df_filtered)} questions to {output_file}")
print(f"Original file {input_file} remains unchanged")

