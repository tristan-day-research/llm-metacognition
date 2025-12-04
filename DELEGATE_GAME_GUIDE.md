# Delegate Game Guide

## Differences Between the Three Files

### 1. `delegate_game_from_capabilities.py` (Original)
- **Decision Mode**: `DECISION_ONLY = True` (digits-only: choose 1/2 or A/D)
- **Teammate Accuracy**: 0.8 (80%)
- **Dataset**: GPQA
- **Model**: llama-3.1-8b-instruct
- **Phase 1 Context**: Summary only (`USE_PHASE1_SUMMARY = True`, `USE_PHASE1_HISTORY = False`)
- **Decision Options**: ["A", "D"] or ["1", "2"] (configurable via `alternate_decision_mapping`)

### 2. `delegate_game_from_capabilities copy.py`
- **Decision Mode**: `DECISION_ONLY = False` (standard: answer with A/B/C/D or delegate with T)
- **Teammate Accuracy**: 0.3 (30%)
- **Dataset**: SimpleMC
- **Model**: kimi-k2
- **Phase 1 Context**: Full history (`USE_PHASE1_SUMMARY = False`, `USE_PHASE1_HISTORY = True`)
- **Decision Options**: Hardcoded ["1", "2"] for decision-only mode

### 3. `delegate_game_from_capabilities copy 2.py` (Currently Open)
- **Decision Mode**: `DECISION_ONLY = True`
- **Teammate Accuracy**: 0.5 (50%)
- **Dataset**: SimpleMC
- **Model**: deepseek-chat
- **Phase 1 Context**: Summary only (`USE_PHASE1_SUMMARY = True`, `USE_PHASE1_HISTORY = False`)
- **Decision Options**: ["A", "D"] or ["1", "2"] (configurable)

## How to Run the Game

### Prerequisites
1. **Capabilities File**: You need a Phase 1 completed results file. The game looks for it in:
   - `./compiled_results_smc/{MODEL_NAME}_phase1_compiled.json` (for SimpleMC)
   - `./compiled_results_sqa/{MODEL_NAME}_phase1_compiled.json` (for SimpleQA)
   - `./completed_results_{DATASET}/{MODEL_NAME}_phase1_completed.json` (for other datasets)

2. **Model Access**: The model must be accessible via OpenRouter API (or other configured provider)

### Steps to Run

1. **Choose a file** (recommend `delegate_game_from_capabilities copy 2.py` for SimpleMC)

2. **Edit the `main()` function** to set your model and dataset:
```python
def main():
    DATASETS = ["SimpleMC"]  # or "GPQA", "SimpleQA", etc.
    models = ["your-model-name"]  # Your fine-tuned model name
    for model in models:
        for d in DATASETS:
            real_main(model, d)
```

3. **Adjust parameters in `real_main()`** if needed:
   - `DECISION_ONLY`: True for decision-only mode, False for full answers
   - `TEAMMATE_ACCURACY_PHASE1` and `TEAMMATE_ACCURACY_PHASE2`: Teammate performance
   - `N_TRIALS_PHASE1` and `N_TRIALS_PHASE2`: Number of questions
   - `USE_PHASE1_SUMMARY` and `USE_PHASE1_HISTORY`: How Phase 1 context is provided

4. **Run the script**:
```bash
python "delegate_game_from_capabilities copy 2.py"
```

5. **Results** will be saved to `delegate_game_logs/` directory

## Using Your Fine-Tuned Model

The Delegate Game currently uses API calls (OpenRouter, Anthropic, etc.) via `BaseGameClass`. To use your local fine-tuned model, you have **three options**:

### Option 1: Use OpenRouter (Easiest)
If your fine-tuned model is available on OpenRouter or HuggingFace (and accessible via OpenRouter), you can use it directly by setting the model name in `main()`.

### Option 2: Add Local Model Support to BaseGameClass (Recommended)
Modify `base_game_class.py` to support local transformers models:

1. Add a "Local" provider option
2. Load your fine-tuned model using `load_finetuned_model()` from `finetune_load_finetuned_model.py`
3. Implement local inference in `_get_llm_answer()` method

**Example modification needed:**
```python
# In BaseGameClass.__init__ or _setup_provider
if self.subject_name.startswith("local-") or self.provider == "Local":
    from finetune_load_finetuned_model import load_finetuned_model
    # Extract model path from subject_name (e.g., "local-/path/to/model")
    model_path = self.subject_name.replace("local-", "")
    self.model, self.tokenizer = load_finetuned_model(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_repo=model_path,
        merge=True
    )
    self.provider = "Local"
```

Then add local inference logic in `_get_llm_answer()`.

### Option 3: Use Local Inference Server
Run a local inference server (vLLM, text-generation-inference, or llama.cpp server) and point OpenRouter to it, or modify the code to use the local server's API.

## Recommended File for Your Use Case

For testing metacognition generalization with a fine-tuned model, I recommend using **`delegate_game_from_capabilities copy 2.py`** because:
- It uses `DECISION_ONLY = True` which tests pure decision-making
- It's configured for SimpleMC (common dataset)
- It has flexible decision options
- It uses summary mode (cleaner for testing)

## Key Parameters to Understand

- **`DECISION_ONLY`**: If True, model only chooses Answer/Delegate (no actual answer needed). If False, model provides full answers or delegates.
- **`TEAMMATE_ACCURACY_PHASE1/PHASE2`**: Controls how good the simulated teammate is (affects delegation strategy)
- **`USE_PHASE1_SUMMARY`**: If True, provides summary stats. If False, no summary.
- **`USE_PHASE1_HISTORY`**: If True, provides full conversation history from Phase 1. If False, no history.
- **`alternate_decision_mapping`**: If True, alternates which digit means Answer/Delegate each trial (reduces position bias)

## Next Steps

1. Ensure you have a capabilities file (Phase 1 results) for your fine-tuned model
2. Choose which file to use (recommend copy 2)
3. Modify `main()` to use your model name
4. If using a local model, implement Option 2 above
5. Run and analyze results in `delegate_game_logs/`

