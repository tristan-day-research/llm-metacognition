"""
test_probes.py

Quick test to verify probe training works before full run.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

print("Testing probe training on synthetic data...")

# Create synthetic data (smaller for testing)
np.random.seed(42)
n_train = 100
n_test = 20
n_features = 100  # Smaller for test

# Synthetic activations (use float32 for numerical stability)
X_train = np.random.randn(n_train, n_features).astype(np.float32)
X_test = np.random.randn(n_test, n_features).astype(np.float32)

# Synthetic entropy (linear combination of first 10 features + noise)
true_weights = np.zeros(n_features)
true_weights[:10] = np.random.randn(10) * 0.5
y_train = X_train @ true_weights + 0.1 * np.random.randn(n_train)
y_test = X_test @ true_weights + 0.1 * np.random.randn(n_test)

# Train probe
probe = Ridge(alpha=0.1)
probe.fit(X_train, y_train)

# Evaluate
y_pred = probe.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✓ Probe trained successfully")
print(f"  R² on test set: {r2:.3f}")
print(f"  (Should be ~0.2-0.4 for synthetic data)")

if r2 > 0.1:
    print("\n✓ All checks passed! Ready to run Day 1 analysis.")
else:
    print("\n⚠️  R² is very low. This might indicate an issue.")
