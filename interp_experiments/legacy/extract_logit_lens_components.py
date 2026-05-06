"""Extract per-model logit-lens components (W_U[option_ids] + final norm weight).

Run on the remote machine where the models are downloaded. Produces tiny .npz
files (~20 KB each) that are pulled locally for use in Section 12 of
analyze_activations.ipynb (logit-lens / depth dynamics analysis).

Usage:
    python extract_logit_lens_components.py

Outputs are written to outputs/logit_lens_components_{model_short}.npz
with keys:
    W_U:         (4, d_model) — unembed rows for tokens [A, B, C, D]
    final_norm:  (d_model,)   — final RMSNorm weight (pre-LM-head)
    option_ids:  (4,)         — token IDs for A, B, C, D (sanity check)
    option_strs: (4,)         — the strings ['A', 'B', 'C', 'D'] decoded back
    model_name:  str          — original HF model name (sanity check)

Pull the files locally afterward, e.g.:
    scp remote:workspace/.../outputs/logit_lens_components_*.npz outputs/
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure which models to extract from.
# For a LoRA-adapted finetuned model, the lm_head and final_norm are typically
# inherited from the base instruct model (LoRA touches attention/MLP layers,
# not the unembed). If your adapter DOES modify these, uncomment the adapter
# branch below.
MODELS_TO_EXTRACT = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

# Optional: if you want a separate extraction for a finetuned model (LoRA adapter),
# set these. Otherwise leave as None — the analyze notebook will reuse the
# instruct W_U for the finetuned model.
ADAPTER_PATHS = [
    # ("meta-llama/Llama-3.1-8B-Instruct",
    #  "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"),
]

OPTIONS = ["A", "B", "C", "D"]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def short_name(hf_name: str) -> str:
    return hf_name.split("/")[-1]


def extract(model_name: str, adapter_path: str | None = None) -> Path:
    """Load model, pull out the unembed rows and final norm weight, save as npz."""
    print(f"\n=== Extracting logit-lens components for {model_name} ===")
    if adapter_path:
        print(f"    (with adapter: {adapter_path})")

    tok = AutoTokenizer.from_pretrained(model_name)
    # IMPORTANT: use add_special_tokens=False — we want the raw option token
    option_ids = [tok.encode(opt, add_special_tokens=False)[0] for opt in OPTIONS]

    # Sanity check: decoding the ids should give back the letters
    decoded = [tok.decode([tid]).strip() for tid in option_ids]
    print(f"    Option IDs: {option_ids}  decoded: {decoded}")
    if any(d != OPTIONS[i] for i, d in enumerate(decoded)):
        raise RuntimeError(
            f"Tokenizer produced unexpected option tokens: ids={option_ids}, "
            f"decoded={decoded}. Llama-style tokenizers should give back 'A','B','C','D'."
        )

    # Load model. We only need lm_head.weight and model.norm.weight — both small —
    # but the cleanest way to get them is via the standard HF loader.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    if adapter_path:
        from peft import PeftModel  # only imported when needed
        model = PeftModel.from_pretrained(model, adapter_path)
        # After merging, ensure we access the underlying base correctly
        if hasattr(model, "get_base_model"):
            base = model.get_base_model()
        else:
            base = model.model if hasattr(model, "model") else model
        lm_head = base.lm_head if hasattr(base, "lm_head") else model.lm_head
        final_norm = base.model.norm if hasattr(base, "model") else model.model.norm
    else:
        lm_head = model.lm_head
        final_norm = model.model.norm

    # Extract the 4 rows of W_U and the RMSNorm weight vector
    W_U = lm_head.weight.data[option_ids].float().cpu().numpy()  # (4, d_model)
    final_norm_w = final_norm.weight.data.float().cpu().numpy()  # (d_model,)

    print(f"    W_U shape: {W_U.shape}")
    print(f"    final_norm shape: {final_norm_w.shape}")

    # Build output filename
    stem = short_name(model_name)
    if adapter_path:
        stem = f"{stem}_adapter-{short_name(adapter_path)}"
    out_path = OUTPUT_DIR / f"logit_lens_components_{stem}.npz"

    np.savez(
        out_path,
        W_U=W_U.astype(np.float16),
        final_norm=final_norm_w.astype(np.float32),
        option_ids=np.array(option_ids, dtype=np.int32),
        option_strs=np.array(OPTIONS),
        model_name=np.array([model_name]),
        adapter=np.array([adapter_path or ""]),
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"    ✓ Saved {out_path}  ({size_kb:.1f} KB)")

    # Free GPU memory (these scripts can be chained)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out_path


if __name__ == "__main__":
    for name in MODELS_TO_EXTRACT:
        extract(name, adapter_path=None)
    for base, adapter in ADAPTER_PATHS:
        extract(base, adapter_path=adapter)
    print("\n✓ Done. Pull outputs/logit_lens_components_*.npz to your local machine.")
