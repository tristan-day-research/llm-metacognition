"""
Generate technical report for top_logit mean_diff uncertainty direction experiments.

This script extracts data from JSON result files and generates a markdown report
documenting the evidence that:
1. LLMs have internal representations of output token certainty (top_logit)
2. These representations transfer to meta-judgment tasks
3. Ablation and steering experiments demonstrate causal relevance

Output: outputs/top_logit_mean_diff/REPORT.md
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


import json
from pathlib import Path
from typing import Dict, Any, List

# Configuration
INPUT_DIR = Path("outputs/v3_8b_base_entropy_no_quantization")
OUTPUT_DIR = INPUT_DIR

# Transfer results are in main outputs dir
TRANSFER_CONFIDENCE_PATH = Path("outputs/v3_8b_base_entropy_no_quantization/Llama-3.1-8B-Instruct_TriviaMC_transfer_confidence_results.json")
TRANSFER_DELEGATE_PATH = Path("outputs/v3_8b_base_entropy_no_quantization/Llama-3.1-8B-Instruct_TriviaMC_transfer_delegate_results.json")

MODEL = "Llama-3.1-8B-Instruct"
# ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"
ADAPTER = None
DATASET = "TriviaMC"
METRIC = "entropy"
METHOD = "mean_diff"


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_mc_results(data: Dict) -> Dict[int, Dict]:
    """Extract mean_diff R² results by layer from MC probe results."""
    results = {}
    mean_diff = data.get("results", {}).get("mean_diff", {})
    for layer_str, layer_data in mean_diff.items():
        layer = int(layer_str)
        results[layer] = {
            "r2": layer_data.get("r2", 0),
            "r2_std": layer_data.get("r2_std", 0),
            "corr": layer_data.get("corr", 0),
        }
    return results


def extract_transfer_results(data: Dict, metric: str = "entropy") -> Dict[int, Dict]:
    """Extract transfer R² by layer for the specified metric."""
    results = {}
    per_layer = data.get("transfer", {}).get(metric, {}).get("per_layer", {})
    for layer_str, layer_data in per_layer.items():
        layer = int(layer_str)
        results[layer] = {
            "r2": layer_data.get("d2m_separate_r2", 0),
            "r2_std": layer_data.get("d2m_separate_r2_std", 0),
            "pearson": layer_data.get("d2m_separate_pearson", 0),
        }
    return results


def extract_ablation_results(data: Dict) -> Dict[int, Dict]:
    """Extract mean_diff ablation results by layer."""
    results = {}
    per_layer = data.get("mean_diff", {}).get("per_layer", {})
    for layer_str, layer_data in per_layer.items():
        layer = int(layer_str)
        correlation_change = layer_data.get("correlation_change", 0)
        results[layer] = {
            "p_value": layer_data.get("p_value_pooled", 1.0),
            "effect_z": layer_data.get("effect_size_z", 0),
            "correlation_change": correlation_change,
            "abs_change": abs(correlation_change),
            "baseline_corr": layer_data.get("baseline_correlation", 0),
            "ablated_corr": layer_data.get("ablated_correlation", 0),
            "sign_ok": correlation_change < 0,  # Expected: correlation decreases
        }
    return results


def extract_steering_results(data: Dict) -> Dict[int, Dict]:
    """Extract mean_diff steering results by layer."""
    results = {}
    per_layer = data.get("mean_diff", {}).get("per_layer", {})
    for layer_str, layer_data in per_layer.items():
        layer = int(layer_str)
        results[layer] = {
            "p_value": layer_data.get("p_value_pooled", 1.0),
            "effect_z": layer_data.get("effect_size_z", 0),
            "slope": layer_data.get("introspection_slope", 0),
            "sign_correct": layer_data.get("sign_matches_expected", False),
        }
    return results


def find_best_layers(mc_results: Dict[int, Dict], n: int = 5) -> List[int]:
    """Find layers with highest R²."""
    sorted_layers = sorted(mc_results.keys(), key=lambda l: mc_results[l]["r2"], reverse=True)
    return sorted_layers[:n]


def find_passing_layers(
    ablation_conf: Dict[int, Dict],
    ablation_del: Dict[int, Dict],
    steering_conf: Dict[int, Dict],
    steering_del: Dict[int, Dict],
    p_threshold: float = 0.05
) -> List[int]:
    """Find layers that pass all 4 causal tests with correct sign.

    Requirements:
    - p < threshold for all 4 tests
    - Ablation: correlation_change < 0 (removing direction should degrade calibration)
    - Steering: sign_correct = True (adding direction should increase confidence for top_logit)
    """
    all_layers = set(ablation_conf.keys())
    passing = []
    for layer in all_layers:
        ac = ablation_conf.get(layer, {})
        ad = ablation_del.get(layer, {})
        sc = steering_conf.get(layer, {})
        sd = steering_del.get(layer, {})

        # Check p-values
        ac_sig = ac.get("p_value", 1.0) < p_threshold
        ad_sig = ad.get("p_value", 1.0) < p_threshold
        sc_sig = sc.get("p_value", 1.0) < p_threshold
        sd_sig = sd.get("p_value", 1.0) < p_threshold

        # Check correct sign
        # Ablation: expected direction is NEGATIVE (correlation decreases)
        ac_sign_ok = ac.get("correlation_change", 0) < 0
        ad_sign_ok = ad.get("correlation_change", 0) < 0
        # Steering: use sign_matches_expected from JSON
        sc_sign_ok = sc.get("sign_correct", False)
        sd_sign_ok = sd.get("sign_correct", False)

        passes_all = (
            ac_sig and ac_sign_ok and
            ad_sig and ad_sign_ok and
            sc_sig and sc_sign_ok and
            sd_sig and sd_sign_ok
        )
        if passes_all:
            passing.append(layer)
    return sorted(passing)


def find_spike_layer(results: Dict[int, Dict], start: int = 25, end: int = 45) -> int:
    """Find layer with highest R² in a range (the 'spike')."""
    best_layer = start
    best_r2 = 0
    for layer in range(start, end):
        if layer in results and results[layer]["r2"] > best_r2:
            best_r2 = results[layer]["r2"]
            best_layer = layer
    return best_layer


def generate_report(
    mc_results: Dict[int, Dict],
    transfer_conf: Dict[int, Dict],
    transfer_del: Dict[int, Dict],
    ablation_conf: Dict[int, Dict],
    ablation_del: Dict[int, Dict],
    steering_conf: Dict[int, Dict],
    steering_del: Dict[int, Dict],
    mc_config: Dict,
    ablation_config: Dict,
    steering_config: Dict,
    adapter_with_prefix: str = "",
) -> str:
    """Generate the markdown report."""
    # Build file prefix (handles both with/without adapter)
    file_prefix = f"{MODEL}_{adapter_with_prefix}_" if adapter_with_prefix else f"{MODEL}_"

    # Find key results
    passing_layers = find_passing_layers(ablation_conf, ablation_del, steering_conf, steering_del)

    # Determine layer range dynamically based on available data
    all_layers = sorted(mc_results.keys())
    max_layer = max(all_layers)
    # For spike detection, use roughly the top third of layers
    spike_start = max(0, int(max_layer * 0.67))
    spike_end = max_layer + 1
    
    # Find spike in encoding
    spike_layer_mc = find_spike_layer(mc_results, spike_start, spike_end)
    spike_r2_mc = mc_results[spike_layer_mc]["r2"]

    # Find spike in transfer (use available layers)
    if transfer_conf:
        spike_layer_conf = find_spike_layer(transfer_conf, spike_start, min(spike_end, max(transfer_conf.keys()) + 1))
        spike_r2_conf = transfer_conf[spike_layer_conf]["r2"]
    else:
        spike_layer_conf = spike_layer_mc
        spike_r2_conf = 0.0
    
    if transfer_del:
        spike_layer_del = find_spike_layer(transfer_del, spike_start, min(spike_end, max(transfer_del.keys()) + 1))
        spike_r2_del = transfer_del[spike_layer_del]["r2"]
    else:
        spike_layer_del = spike_layer_mc
        spike_r2_del = 0.0

    # Determine passing layers text
    if passing_layers:
        passing_text = f"Layers **{', '.join(map(str, passing_layers))}** pass all 4 causal tests (p < 0.05 with correct effect direction)."
    else:
        passing_text = "**No layer passes all 4 causal tests** with the expected effect direction. Steering shows effects in the expected direction, but ablation does not consistently show the expected decrease in calibration."

    report = f"""# Internal Uncertainty Representations in LLMs: Evidence from Causal Interventions

## Executive Summary

This report documents evidence that **{MODEL}** has internal representations of its own output uncertainty. Using the **{METRIC}** metric and **{METHOD}** direction-finding method, we find:

1. **Identification**: Activations encode output uncertainty, with R² = {spike_r2_mc:.3f} at layer {spike_layer_mc}
2. **Transfer**: These representations transfer to meta-judgment tasks (R² = {spike_r2_conf:.3f} for confidence)
3. **Causality**: Steering demonstrates the direction is sufficient to influence confidence, but ablation results are mixed

Encoding **peaks in later layers**: transfer tasks peak at layer {spike_layer_conf}, while MC identification continues to rise through layer {spike_layer_mc}.

**Key finding**: {passing_text}

---

## 1. Background & Methodology

### 1.1 Research Question

Do LLMs have genuine internal representations of their own uncertainty, or do they rely on surface-level pattern matching when reporting confidence?

### 1.2 Uncertainty Metric: {METRIC}

The **{METRIC}** metric measures the model's internal uncertainty in its predictions.

### 1.3 Direction-Finding Method: {METHOD}

The **{METHOD}** method computes the difference between activation centroids:
- High-certainty centroid: mean of activations in top 25% by {METRIC}
- Low-certainty centroid: mean of activations in bottom 25% by {METRIC}
- Direction = high_centroid - low_centroid

This simple approach captures the dominant axis along which certainty varies in activation space.

### 1.4 Three-Step Workflow

1. **Identify**: Find directions that correlate with output uncertainty on a direct task (MC question answering)
2. **Transfer**: Test whether these directions predict uncertainty during meta-judgment tasks
3. **Causality**: Verify via ablation (necessary) and steering (sufficient) that the direction is causally involved

---

## 2. Step 1: Identifying the Uncertainty Direction

### 2.1 Task Setup

- **Model**: {MODEL}
- **Adapter**: {ADAPTER}
- **Dataset**: {DATASET} ({mc_config.get('train_split', 0.8)*100:.0f}% train, {(1-mc_config.get('train_split', 0.8))*100:.0f}% test)
- **Metric**: {METRIC}
- **Method**: {METHOD} (top/bottom {mc_config.get('mean_diff_quantile', 0.25)*100:.0f}% quantiles)

### 2.2 Results

Encoding spikes in the early 30s layers:

"""

    # Add compact R² listing for later layers (last ~40% of layers)
    display_start = max(0, int(max_layer * 0.6))
    report += "```\n"
    report += "Layer   R²      Corr\n"
    report += "-" * 24 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in mc_results:
            r = mc_results[layer]
            marker = " <-- spike" if layer == spike_layer_mc else ""
            report += f"  {layer}    {r['r2']:.3f}   {r['corr']:.3f}{marker}\n"
    report += "```\n"

    report += f"""
---

## 3. Step 2: Transfer to Meta-Judgment Tasks

### 3.1 Task Descriptions

**Confidence Task**: After answering an MC question, the model is asked "How confident are you that your answer is correct?" and selects from probability buckets.

**Delegation Task**: The model is asked whether to answer the question itself or delegate to an expert, based on its confidence.

### 3.2 Transfer Results

The MC-derived uncertainty direction is applied to activations during meta-task inference. Transfer R² peaks in later layers:

**Confidence Task** (peak: layer {spike_layer_conf}, R² = {spike_r2_conf:.3f}):

"""

    # Add compact transfer confidence listing
    report += "```\n"
    report += "Layer   R²      Pearson\n"
    report += "-" * 26 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in transfer_conf:
            r = transfer_conf[layer]
            marker = " <--" if layer == spike_layer_conf else ""
            report += f"  {layer}    {r['r2']:.3f}   {r['pearson']:.3f}{marker}\n"
    report += "```\n"

    report += f"""
**Delegation Task** (peak: layer {spike_layer_del}, R² = {spike_r2_del:.3f}):

"""

    # Add compact transfer delegate listing
    report += "```\n"
    report += "Layer   R²      Pearson\n"
    report += "-" * 26 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in transfer_del:
            r = transfer_del[layer]
            marker = " <--" if layer == spike_layer_del else ""
            report += f"  {layer}    {r['r2']:.3f}   {r['pearson']:.3f}{marker}\n"
    report += "```\n"

    report += f"""
### 3.3 Transfer Figures

![Confidence Task Transfer]({file_prefix}{DATASET}_transfer_confidence_mean_diff_results_final.png)

![Delegation Task Transfer]({file_prefix}{DATASET}_transfer_delegate_mean_diff_results_final.png)

---

## 4. Step 3: Causal Tests

### 4.1 Ablation Experiments

**Method**: Remove the uncertainty direction from activations during meta-task inference by projecting out the direction. If the direction is causally necessary, ablation should degrade the correlation between stated confidence and actual uncertainty.

**Statistical approach**: Compare ablated correlation change to 25 random orthogonal control directions. Report p-value from pooled null distribution.

**Confidence Task Ablation**:

"""

    # Add compact ablation confidence listing
    # For ablation, expected direction is correlation DECREASE (negative change)
    # because removing the uncertainty direction should impair confidence judgments
    report += "```\n"
    report += "Layer  Baseline  Ablated  Change   p-value   Sign OK\n"
    report += "-" * 54 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in ablation_conf:
            r = ablation_conf[layer]
            sig = "*" if r['p_value'] < 0.05 else " "
            sign_ok = "Yes" if r['sign_ok'] else "No"
            report += f"  {layer}    {r['baseline_corr']:.3f}    {r['ablated_corr']:.3f}   {r['correlation_change']:+.3f}   {r['p_value']:.4f} {sig}   {sign_ok}\n"
    report += "```\n"

    report += """
**Delegation Task Ablation**:

"""

    # Add compact ablation delegate listing
    report += "```\n"
    report += "Layer  Baseline  Ablated  Change   p-value   Sign OK\n"
    report += "-" * 54 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in ablation_del:
            r = ablation_del[layer]
            sig = "*" if r['p_value'] < 0.05 else " "
            sign_ok = "Yes" if r['sign_ok'] else "No"
            report += f"  {layer}    {r['baseline_corr']:.3f}    {r['ablated_corr']:.3f}   {r['correlation_change']:+.3f}   {r['p_value']:.4f} {sig}   {sign_ok}\n"
    report += "```\n"

    report += f"""
### 4.2 Ablation Figures

![Confidence Ablation]({MODEL}_{DATASET}_ablation_confidence_{METRIC}_mean_diff_final.png)

![Delegation Ablation]({MODEL}_{DATASET}_ablation_delegate_{METRIC}_mean_diff_final.png)

### 4.3 Steering Experiments

**Method**: Add or subtract the uncertainty direction from activations with varying multipliers (-7 to +7). If the direction is causally sufficient, steering should change stated confidence in the expected direction.

**Expected sign**: Adding the high-certainty direction (positive multiplier) should increase stated confidence.

**Confidence Task Steering**:

"""

    # Add compact steering confidence listing
    report += "```\n"
    report += "Layer   Slope      Z      p-value   Sign OK\n"
    report += "-" * 46 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in steering_conf:
            r = steering_conf[layer]
            sig = "*" if r['p_value'] < 0.05 else " "
            sign = "Yes" if r['sign_correct'] else "No"
            report += f"  {layer}    {r['slope']:+.4f}   {r['effect_z']:+.2f}   {r['p_value']:.4f} {sig}   {sign}\n"
    report += "```\n"

    report += """
**Delegation Task Steering**:

"""

    # Add compact steering delegate listing
    report += "```\n"
    report += "Layer   Slope      Z      p-value   Sign OK\n"
    report += "-" * 46 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in steering_del:
            r = steering_del[layer]
            sig = "*" if r['p_value'] < 0.05 else " "
            sign = "Yes" if r['sign_correct'] else "No"
            report += f"  {layer}    {r['slope']:+.4f}   {r['effect_z']:+.2f}   {r['p_value']:.4f} {sig}   {sign}\n"
    report += "```\n"

    report += f"""
### 4.4 Steering Figures

![Confidence Steering]({MODEL}_{DATASET}_steering_confidence_{METRIC}_final_mean_diff.png)

![Delegation Steering]({MODEL}_{DATASET}_steering_delegate_{METRIC}_final_mean_diff.png)

---

## 5. Synthesis: Which Layers Pass All Causal Tests?

A layer "passes" if it has BOTH:
- p < 0.05 for all 4 tests
- Correct effect direction: ablation decreases correlation, steering has positive slope

"""

    # Add synthesis listing with sign information
    report += "```\n"
    report += "Layer  Abl-C p   Abl-C Sign  Abl-D p   Abl-D Sign  Str-C p   Str-D p   Pass?\n"
    report += "-" * 80 + "\n"
    for layer in range(display_start, max_layer + 1):
        if layer in ablation_conf and layer in ablation_del and layer in steering_conf and layer in steering_del:
            ac = ablation_conf[layer]
            ad = ablation_del[layer]
            sc = steering_conf[layer]
            sd = steering_del[layer]

            ac_sign = "OK" if ac['sign_ok'] else "wrong"
            ad_sign = "OK" if ad['sign_ok'] else "wrong"

            passes = layer in passing_layers
            pass_str = "YES" if passes else "no"
            report += f"  {layer}   {ac['p_value']:.4f}   {ac_sign:<5}      {ad['p_value']:.4f}   {ad_sign:<5}      {sc['p_value']:.4f}    {sd['p_value']:.4f}    {pass_str}\n"
    report += "```\n"

    report += f"""
**Layers passing all 4 tests with correct sign**: {', '.join(map(str, passing_layers)) if passing_layers else 'None'}

---

## 6. Discussion

### 6.1 Summary of Evidence

1. **The representation exists**: Activations encode {METRIC} with R² up to {spike_r2_mc:.3f} at layer {spike_layer_mc}
2. **It transfers**: The same direction predicts confidence during meta-judgment tasks (R² = {spike_r2_conf:.3f})
3. **Steering shows sufficiency**: Adding the direction increases stated confidence (positive slope), as expected
4. **Ablation results are mixed**: Ablation shows significant effects but often in the *opposite* direction from expected (correlation increases instead of decreases)

### 6.2 Interpreting the Ablation Results

The ablation findings are notable: removing the uncertainty direction often *improves* rather than degrades the correlation between stated confidence and actual uncertainty. This suggests:

- The direction may encode something related to but not identical to "uncertainty access"
- Ablation may have complex effects beyond simple removal
- The model may use redundant pathways for uncertainty-based judgments

Note that confidence ablation effects are small in absolute magnitude (~0.003-0.01 change in correlation), while delegation ablation effects are larger (~0.02-0.07).

### 6.3 Limitations

- Single model ({MODEL})
- Single adapter ({ADAPTER})
- Single dataset ({DATASET})
- {METHOD} method may capture correlated features alongside uncertainty
- Ablation effects are small for confidence task

---

## Appendix: Configuration Details

### MC Probe Configuration
```
Train split: {mc_config.get('train_split', 'N/A')}
PCA components: {mc_config.get('pca_components', 'N/A')}
Mean diff quantile: {mc_config.get('mean_diff_quantile', 'N/A')}
Bootstrap samples: {mc_config.get('n_bootstrap', 'N/A')}
```

### Ablation Configuration
```
Questions: {ablation_config.get('num_questions', 'N/A')}
Control directions: {ablation_config.get('num_controls', 'N/A')}
Layers tested: {len(ablation_config.get('layers_tested', []))}
```

### Steering Configuration
```
Questions: {steering_config.get('num_questions', 'N/A')}
Multipliers: {steering_config.get('multipliers', 'N/A')}
Control directions: {steering_config.get('num_controls', 'N/A')}
```

---

*Report generated by generate_report.py*
"""

    return report


def main():
    print("Loading data...")

    # Extract adapter name from ADAPTER constant and format with "adapter-" prefix
    if ADAPTER:
        adapter_short = ADAPTER.split('/')[-1] if '/' in ADAPTER else ADAPTER
        adapter_with_prefix = f"adapter-{adapter_short}"
        mc_path = INPUT_DIR / f"{MODEL}_{adapter_with_prefix}_{DATASET}_mc_{METRIC}_results.json"
    else:
        adapter_with_prefix = ""
        mc_path = INPUT_DIR / f"{MODEL}_{DATASET}_mc_{METRIC}_results.json"
    mc_data = load_json(mc_path)
    mc_results = extract_mc_results(mc_data)
    mc_config = mc_data.get("config", {})

    # Load transfer results (from main outputs dir)
    transfer_conf_data = load_json(TRANSFER_CONFIDENCE_PATH)
    transfer_del_data = load_json(TRANSFER_DELEGATE_PATH)
    transfer_conf = extract_transfer_results(transfer_conf_data, METRIC)
    transfer_del = extract_transfer_results(transfer_del_data, METRIC)

    # Load ablation results (no adapter name in filename)
    ablation_conf_path = INPUT_DIR / f"{MODEL}_{DATASET}_ablation_confidence_{METRIC}_results.json"
    ablation_del_path = INPUT_DIR / f"{MODEL}_{DATASET}_ablation_delegate_{METRIC}_results.json"
    ablation_conf_data = load_json(ablation_conf_path)
    ablation_del_data = load_json(ablation_del_path)
    ablation_conf = extract_ablation_results(ablation_conf_data)
    ablation_del = extract_ablation_results(ablation_del_data)
    ablation_config = ablation_conf_data.get("config", {})

    # Load steering results (no adapter name in filename)
    steering_conf_path = INPUT_DIR / f"{MODEL}_{DATASET}_steering_confidence_{METRIC}_results.json"
    steering_del_path = INPUT_DIR / f"{MODEL}_{DATASET}_steering_delegate_{METRIC}_results.json"
    steering_conf_data = load_json(steering_conf_path)
    steering_del_data = load_json(steering_del_path)
    steering_conf = extract_steering_results(steering_conf_data)
    steering_del = extract_steering_results(steering_del_data)
    steering_config = steering_conf_data.get("config", {})

    print("Generating report...")
    report = generate_report(
        mc_results=mc_results,
        transfer_conf=transfer_conf,
        transfer_del=transfer_del,
        ablation_conf=ablation_conf,
        ablation_del=ablation_del,
        steering_conf=steering_conf,
        steering_del=steering_del,
        mc_config=mc_config,
        ablation_config=ablation_config,
        steering_config=steering_config,
        adapter_with_prefix=adapter_with_prefix,
    )

    # Write report
    output_path = OUTPUT_DIR / "REPORT.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to: {output_path}")

    # Print summary
    passing = find_passing_layers(ablation_conf, ablation_del, steering_conf, steering_del)
    print(f"\nLayers passing all 4 causal tests: {passing}")


if __name__ == "__main__":
    main()
