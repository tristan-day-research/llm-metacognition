"""
Quick script to verify baseline correlation between top_logit and stated_confidence
from existing output files.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load top_logit from mc_dataset.json
with open('outputs/Llama-3.3-70B-Instruct_TriviaMC_mc_dataset.json') as f:
    mc_data = json.load(f)

top_logit = np.array([item['top_logit'] for item in mc_data['data']])
entropy = np.array([item['entropy'] for item in mc_data['data']])

print(f"Loaded {len(top_logit)} questions from mc_dataset.json")
print(f"  top_logit: mean={top_logit.mean():.4f}, std={top_logit.std():.4f}")
print(f"  entropy:   mean={entropy.mean():.4f}, std={entropy.std():.4f}")

# Load confidences from transfer activations
for task in ["confidence", "delegate"]:
    npz_path = f'outputs/Llama-3.3-70B-Instruct_TriviaMC_transfer_{task}_activations.npz'
    try:
        acts = np.load(npz_path)
        confidences = acts['confidences']
        print(f"\n=== {task.upper()} TASK ===")
        print(f"Loaded {len(confidences)} confidence values")
        print(f"  stated_{task}: mean={confidences.mean():.4f}, std={confidences.std():.4f}")

        # Full dataset correlations
        r_tl, p_tl = pearsonr(top_logit, confidences)
        rho_tl, _ = spearmanr(top_logit, confidences)

        r_ent, p_ent = pearsonr(-entropy, confidences)
        rho_ent, _ = spearmanr(-entropy, confidences)

        print(f"\nFull dataset (N={len(confidences)}):")
        print(f"  top_logit:  r={r_tl:.4f}, ρ={rho_tl:.4f}")
        print(f"  -entropy:   r={r_ent:.4f}, ρ={rho_ent:.4f}")

        # Test set correlation (80/20 split, seed=42 matching test_meta_transfer.py)
        from sklearn.model_selection import train_test_split
        n = len(confidences)
        indices = np.arange(n)
        _, test_idx = train_test_split(indices, train_size=0.8, random_state=42)

        r_test, _ = pearsonr(top_logit[test_idx], confidences[test_idx])
        r_ent_test, _ = pearsonr(-entropy[test_idx], confidences[test_idx])

        print(f"\nTest set (N={len(test_idx)}, 80/20 split, seed=42):")
        print(f"  top_logit:  r={r_test:.4f}")
        print(f"  -entropy:   r={r_ent_test:.4f}")

        # Random 100-question subsample (matching ablation NUM_QUESTIONS=100)
        # Run multiple seeds to show variance
        print(f"\nRandom 100-question subsamples (like ablation):")
        for seed in [42, 123, 456]:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, 100, replace=False)
            r_sub, _ = pearsonr(top_logit[idx], confidences[idx])
            r_ent_sub, _ = pearsonr(-entropy[idx], confidences[idx])
            print(f"  seed={seed}: top_logit r={r_sub:.4f}, -entropy r={r_ent_sub:.4f}")

    except FileNotFoundError:
        print(f"\n=== {task.upper()} TASK ===")
        print(f"  File not found: {npz_path}")
