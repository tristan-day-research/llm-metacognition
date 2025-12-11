"""
diagnose_data.py

Check if activations are loading correctly.
"""

import numpy as np
from pathlib import Path

print("Checking activation data...")

# Load base model activations
base_file = "interp/activations/llama_3.1_8b_base_responses_1000/mcq/activations.npz"
data = np.load(base_file)

print(f"\nFile: {base_file}")
print(f"Keys in npz: {list(data.keys())}")

acts = data['activations']
print(f"\nActivations shape: {acts.shape}")
print(f"Expected: [32 layers, 1000 questions, 4096 dims]")

# Check if all layers are different
print(f"\nLayer 0 == Layer 31? {np.array_equal(acts[0], acts[31])}")
print(f"Layer 0 == Layer 15? {np.array_equal(acts[0], acts[15])}")

# Check if all questions are different  
print(f"\nQuestion 0 == Question 1 at layer 15? {np.array_equal(acts[15, 0], acts[15, 1])}")

# Check actual values
print(f"\nLayer 15, Question 0, first 10 dims:")
print(acts[15, 0, :10])

print(f"\nLayer 31, Question 0, first 10 dims:")
print(acts[31, 0, :10])

# Check if any layer is all zeros
for layer in [0, 15, 31]:
    is_zero = np.all(acts[layer] == 0)
    print(f"Layer {layer} all zeros? {is_zero}")

# Check dtype
print(f"\ndtype: {acts.dtype}")

# Load with the data loader
print("\n" + "="*60)
print("Testing load_collected_data_v2.py...")
print("="*60)

from interp_load_collected_data_v2 import ActivationDataset

dataset = ActivationDataset("interp/activations/llama_3.1_8b_base_responses_1000/mcq")

# Get layer 15
layer_15 = dataset.get_layer_activations(15)
print(f"\nLoaded layer 15 shape: {layer_15.shape}")
print(f"Expected: [1000, 4096]")

# Check if it matches raw npz
print(f"\nDoes loaded data match raw npz?")
print(f"  {np.array_equal(layer_15, acts[15])}")

# Get entropy
entropy = dataset.get_entropy_values()
print(f"\nEntropy shape: {entropy.shape}")
print(f"Entropy stats: mean={entropy.mean():.3f}, std={entropy.std():.3f}, min={entropy.min():.3f}, max={entropy.max():.3f}")

# Check correlation
print(f"\n" + "="*60)
print("Checking if activations correlate with entropy...")
print("="*60)

from scipy.stats import pearsonr

for layer in [0, 10, 20, 31]:
    acts_layer = dataset.get_layer_activations(layer)
    
    # Compute correlation between first activation dimension and entropy
    corr, pval = pearsonr(acts_layer[:, 0], entropy)
    
    print(f"Layer {layer}, dim 0 vs entropy: r={corr:.4f}, p={pval:.4f}")
    
    # Try mean activation
    mean_act = acts_layer.mean(axis=1)
    corr_mean, pval_mean = pearsonr(mean_act, entropy)
    print(f"Layer {layer}, mean activation vs entropy: r={corr_mean:.4f}, p={pval_mean:.4f}")

print("\n✓ Diagnostic complete")
