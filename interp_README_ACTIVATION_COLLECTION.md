# Activation Collection for Introspection Experiments

This pipeline collects comprehensive activation data from Llama models for interpretability analysis.

## Quick Start

### 1. Test Run (10 questions, base model)
```bash
python collect_activations.py \
    --model_type base \
    --num_questions 10 \
    --output_dir test_run_base \
    --batch_size 4
```

### 2. Test Run (10 questions, finetuned model)
```bash
python collect_activations.py \
    --model_type finetuned \
    --lora_checkpoint ckpt_step_1280 \
    --num_questions 10 \
    --output_dir test_run_finetuned \
    --batch_size 4
```

### 3. Full Run (1000+ questions)
```bash
# Base model
python collect_activations.py \
    --model_type base \
    --num_questions 1000 \
    --output_dir full_run_base \
    --batch_size 8

# Finetuned model
python collect_activations.py \
    --model_type finetuned \
    --lora_checkpoint ckpt_step_1280 \
    --num_questions 1000 \
    --output_dir full_run_finetuned \
    --batch_size 8
```

## Data Inspection

### Basic inspection
```bash
python inspect_collected_data.py \
    --data_dir test_run_base/mcq \
    --show_examples 3
```

### Compare base vs finetuned
```bash
python inspect_collected_data.py \
    --data_dir full_run_base/mcq \
    --compare_with full_run_finetuned/mcq \
    --show_examples 5
```

## Output Structure

```
output_dir/
├── run_config.json                    # Run configuration
├── mcq/
│   ├── metadata.json                  # All question metadata (no activations)
│   ├── activations_layer_00.npy       # Layer 0 activations [n_questions, seq_len, 4096]
│   ├── activations_layer_01.npy
│   └── ...                            # Layers 0-31
├── self_conf/
│   ├── metadata.json
│   └── activations_layer_*.npy
└── other_conf/
    ├── metadata.json
    └── activations_layer_*.npy
```

## What Gets Collected

### For Each Question in Each Pass (MCQ, Self-Confidence, Other-Confidence):

**Question Metadata:**
- `question_id` - Unique identifier
- `question_text` - Full question text
- `options` - Dict with A/B/C/D options
- `correct_answer_letter` - Ground truth (MCQ only)

**Prompt & Execution:**
- `model_name` - Model identifier
- `checkpoint_id` - Checkpoint/version
- `prompt_type` - "mcq", "self_confidence", or "other_confidence"
- `prompt_text` - Full formatted prompt
- `temperature`, `top_p`, `max_new_tokens` - Generation params

**Model Output:**
- `output_text` - Raw output
- `parsed_answer` - Extracted answer (A/B/C/D or confidence bin)
- `is_correct` - Boolean (MCQ only)

**Token-Level Distributions:**
- `logits` - Logits over answer choices [4] for MCQ, [8] for confidence
- `probs` - Softmax probabilities
- `entropy` - Distribution entropy
- `logit_margin` - Difference between correct and best incorrect (MCQ only)
- `answer_token_index` or `confidence_token_index` - Position in sequence

**Derived Values:**
- `self_confidence` - Expected confidence 0-100% (self-confidence pass)
- `other_confidence` - Expected confidence 0-100% (other-confidence pass)

**Surface Features:**
- `char_length` - Question length in characters
- `token_length` - Question length in tokens
- `question_type` - what/where/when/who/how/why
- `has_negation` - Boolean
- `has_ambiguity` - Boolean (contains "possibly", "might", etc.)
- `token_rarity_mean` - Mean token ID (proxy for rarity)
- `token_rarity_max` - Max token ID

**Activations:**
- Stored separately as numpy arrays (float16)
- Shape: `[num_questions, seq_length, 4096]` per layer
- 32 layers total (layer 0-31)
- Includes ALL token positions, not just final token

## Memory & Storage

### GPU Memory:
- ~12-16GB for base model
- ~14-18GB for finetuned model with LoRA
- Activations are moved to CPU immediately to save GPU memory

### Storage:
- **Test run (10 questions):** ~50-100 MB per model
- **Full run (1000 questions):** ~3-5 GB per model
- Activations stored in float16 to save space

### Timing (on 40GB A100):
- MCQ pass: ~15-20 min for 1000 questions
- Self-conf pass: ~15-20 min for 1000 questions  
- Other-conf pass: ~15-20 min for 1000 questions
- **Total: ~1-1.5 hours per model** (base + finetuned = 2-3 hours)

## Tips

### 1. Always Test First
Run with `--num_questions 10` before committing to full run:
```bash
python collect_activations.py --model_type base --num_questions 10 --output_dir test
python inspect_collected_data.py --data_dir test/mcq --show_examples 3
```

### 2. Batch Size Selection
- **Large GPU (40-80GB):** `--batch_size 16`
- **Medium GPU (24-40GB):** `--batch_size 8`
- **Small GPU (16-24GB):** `--batch_size 4`

### 3. Intermediate Saves
Script automatically saves every 10 batches to prevent data loss.

### 4. Monitor Progress
The script shows progress bars and saves incrementally. You can safely Ctrl+C and resume.

## Next Steps: Analysis

After collecting activations, you can:

1. **Day 1: Layerwise Probing**
   - Train linear probes on each layer
   - Visualize emergence of entropy signal
   - Test temporal lag hypothesis

2. **Day 2: Surface vs Internal**
   - Use collected surface features
   - Compare probe strength on internal vs surface features

3. **Day 3: Confidence Directions**
   - Extract high/low confidence examples
   - Compute contrastive directions

4. **Day 4: Self vs Other**
   - Compare activations across confidence types
   - Test representation similarity

5. **Day 5: Causal Patching**
   - Use collected activations for intervention experiments

## Troubleshooting

### Out of Memory (GPU)
- Reduce `--batch_size`
- Model already moves activations to CPU, but inference needs GPU space

### Out of Memory (CPU/RAM)
- Reduce `--num_questions` per run
- Process in multiple runs and merge later

### Model Not Found
```bash
# For local checkpoints, use full path:
python collect_activations.py \
    --model_type finetuned \
    --lora_checkpoint local_checkpoints/2025-12-03-22-26-59_checkpoints/ckpt_step_1280
```

### Verify Collection Worked
```bash
# Check metadata
python inspect_collected_data.py --data_dir output_dir/mcq

# Check activations exist
ls -lh output_dir/mcq/activations_layer_*.npy

# Check sizes match
python -c "import numpy as np; print(np.load('output_dir/mcq/activations_layer_00.npy').shape)"
```

## Data Format Details

### Activation Arrays
```python
import numpy as np

# Load layer 15 activations for MCQ pass
acts = np.load("output_dir/mcq/activations_layer_15.npy")
# Shape: [num_questions, seq_length, 4096]
# dtype: float16

# Access question 42's activations at layer 15
question_42_acts = acts[42]  # [seq_length, 4096]

# Access final token activation for question 42
final_token_act = acts[42, -1]  # [4096]
```

### Metadata JSON
```python
import json

with open("output_dir/mcq/metadata.json") as f:
    metadata = json.load(f)

# Access question 42's metadata
q42 = metadata[42]
print(q42["question_text"])
print(q42["parsed_answer"])
print(q42["is_correct"])
print(q42["entropy"])
```

## Contact

Questions? Issues? Reach out or check the code comments for details.
