# Issues Analysis: Implicit vs Explicit Confidence Correlation

## Summary of Issues Found

### ✅ Issue 1: Implicit Confidence Computation - **CRITICAL ISSUE FOUND**

**Problem**: In `base_game_class.py` line 334, probabilities are computed by exponentiating logprobs without applying softmax normalization:

```python
probs = [math.exp(tl.logprob) for tl in entry.top_logprobs]
token_probs = dict(zip(tokens, probs))
```

**Why this is wrong**: 
- The logprobs from the API are typically normalized, but when you only have top-k logprobs (not all tokens), exponentiating them directly doesn't guarantee they sum to 1.
- The commented-out code (lines 337-339) shows the correct approach using softmax.

**Impact**: 
- `chosen_prob` may not be a proper probability (doesn't sum to 1 with other options)
- This can distort correlations with explicit confidence
- Entropy calculations may also be affected

**Fix Required**: Apply softmax normalization:
```python
import numpy as np
logprobs = [tl.logprob for tl in entry.top_logprobs]
probs = np.exp(logprobs - np.max(logprobs))  # Numerical stability
probs = probs / probs.sum()  # Normalize to sum to 1
token_probs = dict(zip(tokens, probs))
```

**Status**: The correlation direction appears correct (positive for p(correct), negative for entropy), but the magnitude may be distorted.

---

### ✅ Issue 2: Alignment Between Implicit and Explicit Confidence - **CORRECT**

**Verification**: 
- In `explicit_confidence_task.py` lines 336-344, `probs` are extracted from the direct question prompt
- Lines 361-369 extract `confidence_probs` from the confidence prompt  
- Both are stored in the same `result_dict` with the same question ID (lines 422-425)
- The notebook iterates by question ID, so alignment is correct

**Status**: ✅ No issues found - alignment is correct

---

### ✅ Issue 3: Using Midpoints vs Ordinal Values - **CORRECT**

**Verification**:
- In the notebook Cell 1, `confidence_midpoints` are loaded from `data['run_parameters']['nested_range_midpoints']`
- `confidence_midpoint = confidence_midpoints[confidence_answer]` uses the midpoint value (e.g., 0.85 for "G")
- `expected_confidence` uses weighted average of midpoints: `sum(confidence_midpoints[level] * prob for level, prob in confidence_probs.items())`

**Status**: ✅ Correctly using midpoints, not ordinal values

---

### ✅ Issue 4: Using Correct Logprobs - **CORRECT**

**Verification**:
- `probs` comes from the direct question prompt (line 336 in `explicit_confidence_task.py`)
- `confidence_probs` comes from the confidence prompt (line 361)
- The notebook correctly uses `probs` for implicit confidence and `confidence_probs` for explicit confidence
- Both use empty message history, so they're independent

**Status**: ✅ Correctly using logprobs from question prompt for implicit confidence

---

### ⚠️ Issue 5: Masking Invalid Answers - **POTENTIAL ISSUE**

**Problem**: 
- `_parse_subject_decision` (lines 144-153) tries to extract a single letter, but if it fails, it returns the original answer
- If a model outputs something like "I'd say around 60-80%" or "Maybe B or C", this could:
  1. Not be parsed correctly
  2. Cause a KeyError in the notebook when accessing `confidence_midpoints[confidence_answer]`
  3. Inflate/deflate correlations if invalid answers are included

**Current behavior**:
- The function checks if first or last character is in options
- If not, returns the original string
- No validation that `confidence_answer` is actually a valid bucket (A-H)

**Fix Required**: Add validation in the notebook:
```python
for q_id, q_data in data['results'].items():
    # ... existing code ...
    confidence_answer = q_data['confidence_answer']
    
    # Validate confidence_answer is a valid bucket
    if confidence_answer not in confidence_midpoints:
        print(f"WARNING: Invalid confidence_answer '{confidence_answer}' for question {q_id}")
        continue  # Skip this question or handle appropriately
    
    confidence_midpoint = confidence_midpoints[confidence_answer]
```

**Status**: ⚠️ Potential issue - should add validation

---

## Recommendations

1. **CRITICAL**: Fix softmax normalization in `base_game_class.py` for proper probability computation
2. **IMPORTANT**: Add validation for invalid confidence answers in the notebook
3. **VERIFY**: Check if any questions were skipped due to invalid answers in your current data
4. **RECOMPUTE**: After fixing softmax, recompute all correlations to ensure accuracy

## Correlation Direction Check

Your current correlations:
- `chosen_prob` vs `expected_confidence`: r = 0.263 (positive) ✅ Correct
- `answer_entropy` vs `expected_confidence`: r = -0.330 (negative) ✅ Correct

These directions are correct, but the magnitudes may change after fixing the softmax issue.

