"""

Try multiple probe targets to find what IS encoded in activations.

Tests:
1. Entropy prediction (original)
2. Answer classification (A/B/C/D) 
3. Correctness prediction (correct/incorrect)
4. Confidence prediction (for confidence conditions)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, accuracy_score, classification_report, 
    precision_recall_fscore_support, confusion_matrix,
    mean_absolute_error, mean_squared_error
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from interp_load_collected_data_v2 import ActivationDataset


def train_entropy_probe(X_train, y_train, X_test, y_test, alpha=0.1):
    """Train regression probe for entropy/logit_margin."""
    # Standardize (use float32 for faster computation on M2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.astype(np.float32))
    
    # Train (Ridge is already efficient, but we can use n_jobs if available)
    probe = Ridge(alpha=alpha, solver='auto')  # auto chooses best solver
    probe.fit(X_train_scaled, y_train.astype(np.float32))
    
    # Evaluate
    y_pred_train = probe.predict(X_train_scaled)
    y_pred_test = probe.predict(X_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mse_test': mse_test,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'predictions_train': y_pred_train.tolist(),
        'predictions_test': y_pred_test.tolist(),
    }


def train_classification_probe(X_train, y_train, X_test, y_test, alpha=0.1):
    """Train classification probe (for answer choice or correctness)."""
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.astype(np.float32))
    
    # Train (use 'lbfgs' solver which is faster for small-medium datasets on CPU/M2)
    probe = LogisticRegression(C=1.0/alpha, max_iter=1000, random_state=42, solver='lbfgs', n_jobs=1)
    probe.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_train = probe.predict(X_train_scaled)
    y_pred_test = probe.predict(X_test_scaled)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='weighted', zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred_test)
    
    return {
        'accuracy_train': acc_train,
        'accuracy_test': acc_test,
        'precision_test': precision,
        'recall_test': recall,
        'f1_test': f1,
        'confusion_matrix': cm.tolist(),
        'predictions_train': y_pred_train.tolist(),
        'predictions_test': y_pred_test.tolist(),
    }


def extract_labels(dataset, indices, label_type='entropy', question_type='mcq', mcq_dataset=None):
    """
    Extract different types of labels from dataset.
    
    Probe types:
    - 'entropy': Continuous value - entropy of the model's probability distribution
    - 'answer': Categorical - which letter the model chose
      * For MCQ: A/B/C/D → 0/1/2/3 (4 classes)
      * For self/other: A/B/C/D/E/F/G/H → 0/1/2/3/4/5/6/7 (8 classes, confidence bins)
    - 'correctness': Binary (0/1) - whether the model's answer was correct (1) or incorrect (0)
      * For self/other: looks up is_correct from MCQ dataset using question_id
    - 'confidence': Continuous value - confidence score (for confidence conditions)
    
    Args:
        question_type: 'mcq', 'self', or 'other'
        mcq_dataset: Optional ActivationDataset for MCQ (used to look up is_correct for self/other)
    """
    metadata = dataset.get_metadata(indices)
    
    if label_type == 'entropy':
        # Entropy of the probability distribution over answer choices
        # Higher entropy = more uncertain, lower entropy = more confident
        return np.array([m['entropy'] for m in metadata])
    
    elif label_type == 'logit_margin':
        # Logit margin: difference between top-2 logits
        # Higher margin = more confident, lower margin = less confident
        margins = []
        for m in metadata:
            logits = np.array(m["logits"], dtype=np.float32)
            # assume logits are for [A, B, C, D]
            top2 = np.sort(logits)[-2:]
            margin = float(top2[-1] - top2[-2])
            margins.append(margin)
        return np.array(margins)
    
    elif label_type == 'answer':
        # Which answer choice the model selected
        if question_type == 'mcq':
            # MCQ: A/B/C/D → 0/1/2/3
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        else:
            # Self/Other: A/B/C/D/E/F/G/H → 0/1/2/3/4/5/6/7 (confidence bins)
            answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
        
        return np.array([answer_map[m['parsed_answer']] for m in metadata])
    
    elif label_type == 'correctness':
        # Whether the model's answer was correct (1) or incorrect (0)
        # This is a binary classification task
        if question_type == 'mcq':
            # For MCQ, is_correct is directly in metadata
            if 'is_correct' not in metadata[0]:
                raise ValueError(f"'is_correct' field not found in MCQ metadata.")
            return np.array([int(m['is_correct']) for m in metadata])
        else:
            # For self/other, look up is_correct from MCQ dataset using question_id
            if mcq_dataset is None:
                raise ValueError(f"For {question_type} question type, mcq_dataset must be provided to extract correctness labels.")
            
            # Build lookup dictionary from MCQ dataset
            mcq_metadata = mcq_dataset.get_metadata()
            mcq_lookup = {m['question_id']: m.get('is_correct', None) for m in mcq_metadata}
            
            # Extract correctness for each question_id
            correctness_values = []
            missing_ids = []
            for m in metadata:
                qid = m['question_id']
                if qid in mcq_lookup:
                    is_correct = mcq_lookup[qid]
                    if is_correct is not None:
                        correctness_values.append(int(is_correct))
                    else:
                        missing_ids.append(qid)
                else:
                    missing_ids.append(qid)
            
            if missing_ids:
                print(f"  ⚠️  Warning: {len(missing_ids)} questions missing is_correct in MCQ dataset")
                if len(missing_ids) > len(metadata) * 0.1:  # More than 10% missing
                    raise ValueError(f"Too many questions ({len(missing_ids)}) missing is_correct. Cannot train correctness probe.")
            
            if len(correctness_values) != len(metadata):
                # Filter indices to only those with valid correctness values
                # valid_indices should be positions in the input indices list (which correspond to original dataset indices)
                valid_indices = [i for i, m in enumerate(metadata) 
                               if m['question_id'] in mcq_lookup and mcq_lookup[m['question_id']] is not None]
                if len(valid_indices) < len(metadata) * 0.9:  # Less than 90% valid
                    raise ValueError(f"Too few valid correctness labels ({len(valid_indices)}/{len(metadata)}). Cannot train correctness probe.")
                # Return original dataset indices, not metadata array indices
                original_valid_indices = [indices[i] for i in valid_indices]
                return np.array(correctness_values), original_valid_indices
            
            return np.array(correctness_values)
    
    elif label_type == 'confidence':
        # For confidence conditions
        if 'self_confidence' in metadata[0]:
            return np.array([m['self_confidence'] for m in metadata])
        elif 'other_confidence' in metadata[0]:
            return np.array([m['other_confidence'] for m in metadata])
        else:
            raise ValueError("No confidence labels found")
    
    else:
        raise ValueError(f"Unknown label type: {label_type}")


def save_results_incremental(results, output_dir, model_name, question_type):
    """Save results incrementally after each probe type."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    try:
        results_json = convert_to_json_serializable(results)
        temp_file = output_path / f"results_{model_name}_{question_type}_temp.json"
        final_file = output_path / f"results_{model_name}_{question_type}.json"
        
        with open(temp_file, "w") as f:
            json.dump(results_json, f, indent=2)
        
        # Atomic write: rename temp to final
        temp_file.replace(final_file)
        print(f"  💾 Saved incremental results: {final_file}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not save incremental results: {e}")


def train_all_probes_for_condition(dataset, train_indices, test_indices, question_type='mcq', mcq_dataset=None, output_dir=None, model_name=None, regression_metric='r2', probe_types=None, confidence_target='entropy'):
    """
    Train all probe types for a question type.
    
    Args:
        dataset: ActivationDataset for the current question type
        train_indices: List of training indices
        test_indices: List of test indices
        question_type: 'mcq', 'self', or 'other'
        mcq_dataset: Optional ActivationDataset for MCQ (needed for correctness probe in self/other)
        output_dir: Optional output directory for incremental saving
        model_name: Optional model name for incremental saving
        regression_metric: 'r2', 'mse', or 'both' - metric to use for regression probes
        probe_types: List of probe types to train. If None, uses default based on question_type.
                     Options: 'entropy', 'answer', 'correctness'
    """
    
    results = {
        'question_type': question_type,
        'num_train': len(train_indices),
        'num_test': len(test_indices),
    }
    
    # Determine which probe types to use
    if probe_types is None:
        # Default: use all probes with confidence_target for regression
        if question_type == 'mcq':
            probe_types = [confidence_target, 'answer', 'correctness']
        elif question_type in ['self', 'other']:
            # For self/other: confidence_target, answer (confidence bin), and correctness (from MCQ lookup)
            probe_types = [confidence_target, 'answer', 'correctness']
        else:
            raise ValueError(f"Unknown question_type: {question_type}. Must be 'mcq', 'self', or 'other'")
    
    # Validate probe types
    valid_probe_types = ['entropy', 'logit_margin', 'answer', 'correctness']
    for probe_type in probe_types:
        if probe_type not in valid_probe_types:
            raise ValueError(f"Invalid probe type: {probe_type}. Must be one of {valid_probe_types}")
    
    # Check if correctness probe is requested for self/other without MCQ dataset
    if 'correctness' in probe_types and question_type in ['self', 'other'] and mcq_dataset is None:
        print(f"  ⚠️  Warning: Correctness probe requested for {question_type} but MCQ dataset not provided.")
        print(f"     Skipping correctness probe. Provide MCQ dataset to enable it.")
        probe_types = [p for p in probe_types if p != 'correctness']
    
    # Train probes for each type at each layer
    for probe_type in probe_types:
        print(f"\nTraining {probe_type} probes...")
        
        try:
            # Extract labels
            train_result = extract_labels(dataset, train_indices, probe_type, question_type, mcq_dataset)
            test_result = extract_labels(dataset, test_indices, probe_type, question_type, mcq_dataset)
            
            # Handle case where correctness extraction returns (values, valid_indices)
            # This happens when some questions don't have is_correct in MCQ dataset
            # valid_indices are now the original dataset indices (not positions in the input list)
            if isinstance(train_result, tuple):
                y_train, valid_train_idx = train_result
                train_indices_filtered = valid_train_idx  # Already original dataset indices
                print(f"  Note: Using {len(train_indices_filtered)}/{len(train_indices)} train samples (some missing correctness labels)")
            else:
                y_train = train_result
                train_indices_filtered = train_indices
            
            if isinstance(test_result, tuple):
                y_test, valid_test_idx = test_result
                test_indices_filtered = valid_test_idx  # Already original dataset indices
                print(f"  Note: Using {len(test_indices_filtered)}/{len(test_indices)} test samples (some missing correctness labels)")
            else:
                y_test = test_result
                test_indices_filtered = test_indices
                
        except (KeyError, ValueError) as e:
            print(f"  ⚠️  Skipping {probe_type} probe: {e}")
            continue
        
        # Check if we have enough data
        if len(y_train) < 10 or len(y_test) < 10:
            print(f"  ⚠️  Skipping {probe_type} probe: insufficient data (train: {len(y_train)}, test: {len(y_test)})")
            continue
        
        print(f"  Labels - train: {y_train.shape}, test: {y_test.shape}")
        if probe_type in ['answer', 'correctness']:
            class_dist = np.bincount(y_train)
            print(f"  Class distribution (train): {class_dist}")
            # Store for summary
            results[f'{probe_type}_class_distribution'] = class_dist.tolist()
        else:
            print(f"  Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}")
        
        # Train at each layer
        layer_results = {}
        for layer in tqdm(range(32), desc=f"{probe_type} layers"):
            X_train = dataset.get_layer_activations(layer, train_indices_filtered)
            X_test = dataset.get_layer_activations(layer, test_indices_filtered)
            
            if probe_type in ['entropy', 'logit_margin']:
                result = train_entropy_probe(X_train, y_train, X_test, y_test)
                layer_results[layer] = {
                    'r2_test': result['r2_test'],
                    'r2_train': result['r2_train'],
                    'mse_test': result['mse_test'],
                    'mae_test': result['mae_test'],
                    'rmse_test': result['rmse_test'],
                }
            else:
                result = train_classification_probe(X_train, y_train, X_test, y_test)
                layer_results[layer] = {
                    'accuracy_test': result['accuracy_test'],
                    'accuracy_train': result['accuracy_train'],
                    'precision_test': result['precision_test'],
                    'recall_test': result['recall_test'],
                    'f1_test': result['f1_test'],
                    'confusion_matrix': result['confusion_matrix'],
                }
        
        results[f'{probe_type}_probes'] = layer_results
        
        # Find best layer and emergence layer
        if probe_type in ['entropy', 'logit_margin']:
            # Determine which metric(s) to use
            use_r2 = regression_metric in ['r2', 'both']
            use_mse = regression_metric in ['mse', 'both']
            
            if use_r2:
                # For R², higher is better
                best_layer_r2 = max(layer_results.keys(), key=lambda k: layer_results[k]['r2_test'])
                best_score_r2 = layer_results[best_layer_r2]['r2_test']
                
                # Find emergence layer (first layer where R² > threshold)
                emergence_threshold_r2 = 0.3
                emergence_layer_r2 = None
                for layer in range(32):
                    if layer_results[layer]['r2_test'] > emergence_threshold_r2:
                        emergence_layer_r2 = layer
                        break
                
                results[f'{probe_type}_emergence_layer_r2'] = emergence_layer_r2
                print(f"  Best R² (test): {best_score_r2:.3f} at layer {best_layer_r2}")
                if best_score_r2 < 0:
                    print(f"  ⚠️  WARNING: Negative R² indicates model performs WORSE than predicting the mean!")
                    print(f"     This suggests severe overfitting or data issues. Check train/test split.")
                print(f"  Emergence layer (R² > {emergence_threshold_r2}): {emergence_layer_r2 if emergence_layer_r2 is not None else 'None'}")
            
            if use_mse:
                # For MSE, lower is better
                best_layer_mse = min(layer_results.keys(), key=lambda k: layer_results[k]['mse_test'])
                best_score_mse = layer_results[best_layer_mse]['mse_test']
                
                # Find emergence layer (first layer where MSE < threshold)
                # Use a threshold based on baseline MSE (variance of target)
                baseline_mse = np.var(y_test)
                emergence_threshold_mse = baseline_mse * 0.7  # 30% reduction from baseline
                emergence_layer_mse = None
                for layer in range(32):
                    if layer_results[layer]['mse_test'] < emergence_threshold_mse:
                        emergence_layer_mse = layer
                        break
                
                results[f'{probe_type}_emergence_layer_mse'] = emergence_layer_mse
                print(f"  Best MSE (test): {best_score_mse:.6f} at layer {best_layer_mse}")
                print(f"  Baseline MSE (variance): {baseline_mse:.6f}")
                print(f"  Emergence layer (MSE < {emergence_threshold_mse:.6f}): {emergence_layer_mse if emergence_layer_mse is not None else 'None'}")
            
            # For backward compatibility, also store the primary metric
            if regression_metric == 'r2':
                results[f'{probe_type}_emergence_layer'] = results.get(f'{probe_type}_emergence_layer_r2')
            elif regression_metric == 'mse':
                results[f'{probe_type}_emergence_layer'] = results.get(f'{probe_type}_emergence_layer_mse')
        else:
            best_layer = max(layer_results.keys(), key=lambda k: layer_results[k]['accuracy_test'])
            best_score = layer_results[best_layer]['accuracy_test']
            
            # Find emergence layer (first layer where accuracy > chance + margin)
            if probe_type == 'answer':
                # Chance depends on number of classes
                if question_type == 'mcq':
                    chance = 0.25  # 4 classes (A/B/C/D)
                else:
                    chance = 0.125  # 8 classes (A/B/C/D/E/F/G/H)
            else:
                chance = 0.5  # Binary classification
            emergence_threshold = chance + 0.1  # 10% above chance
            emergence_layer = None
            for layer in range(32):
                if layer_results[layer]['accuracy_test'] > emergence_threshold:
                    emergence_layer = layer
                    break
            
            results[f'{probe_type}_emergence_layer'] = emergence_layer
            print(f"  Best accuracy (test): {best_score:.3f} at layer {best_layer}")
            print(f"  Emergence layer (acc > {emergence_threshold:.2f}): {emergence_layer if emergence_layer is not None else 'None'}")
        
        # Save incrementally after each probe type completes
        if output_dir and model_name:
            save_results_incremental(results, output_dir, model_name, question_type)
    
    return results


def plot_all_probes(results, output_dir, model_name, regression_metric='r2'):
    """Plot results for all probe types."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    question_type = results.get('question_type', results.get('condition', 'unknown'))
    
    # Determine which probes we have
    probe_keys = [k for k in results.keys() if k.endswith('_probes')]
    
    # Count how many plots we need (may need 2 per regression probe if 'both')
    n_plots = 0
    plot_info = []
    for probe_key in probe_keys:
        probe_type = probe_key.replace('_probes', '')
        if probe_type in ['entropy', 'logit_margin']:
            if regression_metric == 'both':
                n_plots += 2
                plot_info.append((probe_key, probe_type, 'r2'))
                plot_info.append((probe_key, probe_type, 'mse'))
            else:
                n_plots += 1
                plot_info.append((probe_key, probe_type, regression_metric))
        else:
            n_plots += 1
            plot_info.append((probe_key, probe_type, None))
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (probe_key, probe_type, metric) in enumerate(plot_info):
        ax = axes[idx]
        
        if probe_type in ['entropy', 'logit_margin']:
            if metric == 'r2':
                scores = [results[probe_key][layer]['r2_test'] for layer in range(32)]
                ax.set_ylabel('R² (Test)', fontsize=12)
                ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Emergence Threshold')
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Baseline (mean)')
                # Dynamic ylim: show full range with some padding
                score_min, score_max = min(scores), max(scores)
                if score_min < 0:
                    # If negative values, show them with padding
                    ylim_min = score_min * 1.1 if score_min < -1 else min(score_min - 0.1, -0.2)
                    ylim_max = max(score_max * 1.1, 1.0) if score_max > 0 else 0.2
                else:
                    ylim_min = -0.2
                    ylim_max = max(score_max * 1.1, 1.0)
                ax.set_ylim(ylim_min, ylim_max)
                title_suffix = ' (R²)'
            else:  # mse
                scores = [results[probe_key][layer]['mse_test'] for layer in range(32)]
                ax.set_ylabel('MSE (Test)', fontsize=12)
                # Dynamic ylim based on MSE range
                score_min, score_max = min(scores), max(scores)
                ylim_min = 0
                ylim_max = score_max * 1.1
                ax.set_ylim(ylim_min, ylim_max)
                title_suffix = ' (MSE)'
            
            # Set title based on probe type
            if probe_type == 'logit_margin':
                probe_title = 'Logit Margin'
            else:
                probe_title = 'Entropy'
        else:
            scores = [results[probe_key][layer]['accuracy_test'] for layer in range(32)]
            ax.set_ylabel('Accuracy (Test)', fontsize=12)
            # Chance level
            if probe_type == 'answer':
                # Determine number of classes from question type
                if question_type == 'mcq':
                    chance = 0.25  # 4 classes (A/B/C/D)
                else:
                    chance = 0.125  # 8 classes (A/B/C/D/E/F/G/H)
            else:
                chance = 0.5   # 2 classes (binary)
            ax.axhline(y=chance, color='red', linestyle='--', alpha=0.5, label=f'Chance ({chance})')
            ax.set_ylim(0, 1.0)
            title_suffix = ''
        
        ax.plot(range(32), scores, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Layer', fontsize=12)
        if probe_type in ['entropy', 'logit_margin']:
            ax.set_title(f'{probe_title} Prediction{title_suffix}\n{model_name} / {question_type}', 
                         fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'{probe_type.title()} Prediction{title_suffix}\n{model_name} / {question_type}', 
                         fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_{question_type}_all_probes.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / f'{model_name}_{question_type}_all_probes.png'}")
    plt.close()


def plot_comparison(base_results, fine_results, output_dir, question_type, regression_metric='r2'):
    """Plot base vs finetuned comparison for all probe types."""
    output_path = Path(output_dir)
    
    probe_keys = [k for k in base_results.keys() if k.endswith('_probes')]
    
    # Count how many plots we need (may need 2 per regression probe if 'both')
    n_plots = 0
    plot_info = []
    for probe_key in probe_keys:
        probe_type = probe_key.replace('_probes', '')
        if probe_type in ['entropy', 'logit_margin']:
            if regression_metric == 'both':
                n_plots += 2
                plot_info.append((probe_key, probe_type, 'r2'))
                plot_info.append((probe_key, probe_type, 'mse'))
            else:
                n_plots += 1
                plot_info.append((probe_key, probe_type, regression_metric))
        else:
            n_plots += 1
            plot_info.append((probe_key, probe_type, None))
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (probe_key, probe_type, metric) in enumerate(plot_info):
        ax = axes[idx]
        
        if probe_type in ['entropy', 'logit_margin']:
            if metric == 'r2':
                base_scores = [base_results[probe_key][layer]['r2_test'] for layer in range(32)]
                fine_scores = [fine_results[probe_key][layer]['r2_test'] for layer in range(32)]
                ylabel = 'R² (Test)'
                threshold = 0.3
                # Dynamic ylim based on actual data range
                all_scores = base_scores + fine_scores
                score_min, score_max = min(all_scores), max(all_scores)
                if score_min < 0:
                    ylim_min = score_min * 1.1 if score_min < -1 else min(score_min - 0.1, -0.2)
                    ylim_max = max(score_max * 1.1, 1.0) if score_max > 0 else 0.2
                else:
                    ylim_min = -0.2
                    ylim_max = max(score_max * 1.1, 1.0)
                ylim = (ylim_min, ylim_max)
                title_suffix = ' (R²)'
                ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Baseline (mean)')
            else:  # mse
                base_scores = [base_results[probe_key][layer]['mse_test'] for layer in range(32)]
                fine_scores = [fine_results[probe_key][layer]['mse_test'] for layer in range(32)]
                ylabel = 'MSE (Test)'
                threshold = None
                all_scores = base_scores + fine_scores
                score_max = max(all_scores)
                ylim = (0, score_max * 1.1)
                title_suffix = ' (MSE)'
        else:
            base_scores = [base_results[probe_key][layer]['accuracy_test'] for layer in range(32)]
            fine_scores = [fine_results[probe_key][layer]['accuracy_test'] for layer in range(32)]
            ylabel = 'Accuracy (Test)'
            if probe_type == 'answer':
                threshold = 0.25 if question_type == 'mcq' else 0.125
            else:
                threshold = 0.5
            ylim = (0, 1.0)
            title_suffix = ''
        
        ax.plot(range(32), base_scores, marker='o', label='Base', linewidth=2, markersize=4)
        ax.plot(range(32), fine_scores, marker='s', label='Finetuned', linewidth=2, markersize=4)
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if probe_type == 'logit_margin':
            probe_title = 'Logit Margin'
        elif probe_type == 'entropy':
            probe_title = 'Entropy'
        else:
            probe_title = probe_type.title()
        ax.set_title(f'{probe_title} Prediction{title_suffix}\n{question_type}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_path / f'comparison_{question_type}_all_probes.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / f'comparison_{question_type}_all_probes.png'}")
    plt.close()


def plot_delta_improvement(base_results, fine_results, output_dir, question_type, regression_metric='r2'):
    """Plot improvement (finetuned - base) across layers."""
    output_path = Path(output_dir)
    
    probe_keys = [k for k in base_results.keys() if k.endswith('_probes')]
    
    # Count how many plots we need (may need 2 per regression probe if 'both')
    n_plots = 0
    plot_info = []
    for probe_key in probe_keys:
        probe_type = probe_key.replace('_probes', '')
        if probe_type in ['entropy', 'logit_margin']:
            if regression_metric == 'both':
                n_plots += 2
                plot_info.append((probe_key, probe_type, 'r2'))
                plot_info.append((probe_key, probe_type, 'mse'))
            else:
                n_plots += 1
                plot_info.append((probe_key, probe_type, regression_metric))
        else:
            n_plots += 1
            plot_info.append((probe_key, probe_type, None))
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    for idx, (probe_key, probe_type, metric) in enumerate(plot_info):
        ax = axes[idx]
        
        if probe_type in ['entropy', 'logit_margin']:
            if metric == 'r2':
                base_scores = np.array([base_results[probe_key][layer]['r2_test'] for layer in range(32)])
                fine_scores = np.array([fine_results[probe_key][layer]['r2_test'] for layer in range(32)])
                ylabel = 'Δ R² (Finetuned - Base)'
                delta = fine_scores - base_scores
            else:  # mse
                base_scores = np.array([base_results[probe_key][layer]['mse_test'] for layer in range(32)])
                fine_scores = np.array([fine_results[probe_key][layer]['mse_test'] for layer in range(32)])
                ylabel = 'Δ MSE (Base - Finetuned)'
                # For MSE, negative delta is improvement (lower is better)
                # So we'll flip the sign for visualization: base - fine means positive = improvement
                delta = base_scores - fine_scores
        else:
            base_scores = np.array([base_results[probe_key][layer]['accuracy_test'] for layer in range(32)])
            fine_scores = np.array([fine_results[probe_key][layer]['accuracy_test'] for layer in range(32)])
            ylabel = 'Δ Accuracy (Finetuned - Base)'
            delta = fine_scores - base_scores
        
        ax.plot(range(32), delta, marker='o', linewidth=2, markersize=4, color='green')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.fill_between(range(32), 0, delta, where=(delta > 0), alpha=0.3, color='green', label='Improvement')
        ax.fill_between(range(32), 0, delta, where=(delta < 0), alpha=0.3, color='red', label='Degradation')
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        title_suffix = f' ({metric.upper()})' if metric else ''
        if probe_type == 'logit_margin':
            probe_title = 'Logit Margin'
        elif probe_type == 'entropy':
            probe_title = 'Entropy'
        else:
            probe_title = probe_type.title()
        ax.set_title(f'Δ {probe_title} Prediction{title_suffix}\n{question_type}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'delta_{question_type}_all_probes.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / f'delta_{question_type}_all_probes.png'}")
    plt.close()


def plot_confusion_matrices(results, output_dir, model_name, question_type):
    """Plot confusion matrices for classification probes at best layer."""
    output_path = Path(output_dir)
    
    probe_keys = [k for k in results.keys() if k.endswith('_probes')]
    classification_probes = [k for k in probe_keys if k.replace('_probes', '') in ['answer', 'correctness']]
    
    if not classification_probes:
        return
    
    n_probes = len(classification_probes)
    fig, axes = plt.subplots(1, n_probes, figsize=(6*n_probes, 5))
    if n_probes == 1:
        axes = [axes]
    
    for idx, probe_key in enumerate(classification_probes):
        ax = axes[idx]
        probe_type = probe_key.replace('_probes', '')
        
        # Find best layer
        best_layer = max(range(32), key=lambda l: results[probe_key][l]['accuracy_test'])
        cm = np.array(results[probe_key][best_layer]['confusion_matrix'])
        
        # Get actual number of classes from confusion matrix shape
        n_classes = cm.shape[0]
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        
        if probe_type == 'answer':
            if question_type == 'mcq':
                # MCQ: 4 classes (A/B/C/D)
                all_labels = ['A', 'B', 'C', 'D']
            else:
                # Self/Other: 8 classes (A/B/C/D/E/F/G/H)
                all_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            
            # Use only the labels that match the confusion matrix size
            labels = all_labels[:n_classes]
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        else:
            # Correctness: binary (2 classes)
            labels = ['Incorrect', 'Correct']
            # Handle case where only one class appears (shouldn't happen, but be safe)
            if n_classes == 1:
                labels = [labels[0]] if cm[0, 0] > 0 else [labels[1]]
            ax.set_xticklabels(labels[:n_classes])
            ax.set_yticklabels(labels[:n_classes])
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(f'{probe_type.title()} Confusion Matrix\nLayer {best_layer} / {model_name}', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / f'{model_name}_{question_type}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path / f'{model_name}_{question_type}_confusion_matrices.png'}")
    plt.close()


def create_summary(base_results, fine_results, output_dir, question_type, regression_metric='r2'):
    """Create text summary of key findings."""
    output_path = Path(output_dir)
    
    summary = []
    summary.append("="*80)
    summary.append("IMPROVED LAYERWISE PROBING - SUMMARY")
    summary.append("="*80)
    summary.append(f"Question Type: {question_type}")
    summary.append(f"Regression Metric: {regression_metric}")
    summary.append("")
    
    probe_keys = [k for k in base_results.keys() if k.endswith('_probes')]
    
    for probe_key in probe_keys:
        probe_type = probe_key.replace('_probes', '')
        summary.append(f"\n{probe_type.upper()} PROBE:")
        summary.append("-" * 80)
        
        # Base model
        if probe_type in ['entropy', 'logit_margin']:
            use_r2 = regression_metric in ['r2', 'both']
            use_mse = regression_metric in ['mse', 'both']
            
            if use_r2:
                base_scores_r2 = [base_results[probe_key][l]['r2_test'] for l in range(32)]
                base_best_layer_r2 = np.argmax(base_scores_r2)
                base_best_score_r2 = base_scores_r2[base_best_layer_r2]
                base_em_r2 = base_results.get(f'{probe_type}_emergence_layer_r2', None)
                summary.append(f"  Base Model (R²):")
                summary.append(f"    Best R²: {base_best_score_r2:.3f} at layer {base_best_layer_r2}")
                summary.append(f"    Emergence layer: {base_em_r2 if base_em_r2 is not None else 'None'}")
            
            if use_mse:
                base_scores_mse = [base_results[probe_key][l]['mse_test'] for l in range(32)]
                base_best_layer_mse = np.argmin(base_scores_mse)
                base_best_score_mse = base_scores_mse[base_best_layer_mse]
                base_em_mse = base_results.get(f'{probe_type}_emergence_layer_mse', None)
                summary.append(f"  Base Model (MSE):")
                summary.append(f"    Best MSE: {base_best_score_mse:.6f} at layer {base_best_layer_mse}")
                summary.append(f"    Emergence layer: {base_em_mse if base_em_mse is not None else 'None'}")
        else:
            base_scores = [base_results[probe_key][l]['accuracy_test'] for l in range(32)]
            base_best_layer = np.argmax(base_scores)
            base_best_score = base_scores[base_best_layer]
            base_em = base_results.get(f'{probe_type}_emergence_layer', None)
            summary.append(f"  Base Model:")
            summary.append(f"    Best Accuracy: {base_best_score:.3f} at layer {base_best_layer}")
            summary.append(f"    Emergence layer: {base_em if base_em is not None else 'None'}")
            # Add class distribution if available
            class_dist = base_results.get(f'{probe_type}_class_distribution', None)
            if class_dist is not None:
                summary.append(f"    Class distribution (train): {class_dist}")
        
        # Finetuned model
        if fine_results:
            if probe_type in ['entropy', 'logit_margin']:
                use_r2 = regression_metric in ['r2', 'both']
                use_mse = regression_metric in ['mse', 'both']
                
                if use_r2:
                    fine_scores_r2 = [fine_results[probe_key][l]['r2_test'] for l in range(32)]
                    fine_best_layer_r2 = np.argmax(fine_scores_r2)
                    fine_best_score_r2 = fine_scores_r2[fine_best_layer_r2]
                    fine_em_r2 = fine_results.get(f'{probe_type}_emergence_layer_r2', None)
                    summary.append(f"  Finetuned Model (R²):")
                    summary.append(f"    Best R²: {fine_best_score_r2:.3f} at layer {fine_best_layer_r2}")
                    summary.append(f"    Emergence layer: {fine_em_r2 if fine_em_r2 is not None else 'None'}")
                    
                    base_em_r2 = base_results.get(f'{probe_type}_emergence_layer_r2', None)
                    if base_em_r2 is not None and fine_em_r2 is not None:
                        shift = fine_em_r2 - base_em_r2
                        summary.append(f"    Emergence shift: {shift:+d} layers")
                
                if use_mse:
                    fine_scores_mse = [fine_results[probe_key][l]['mse_test'] for l in range(32)]
                    fine_best_layer_mse = np.argmin(fine_scores_mse)
                    fine_best_score_mse = fine_scores_mse[fine_best_layer_mse]
                    fine_em_mse = fine_results.get(f'{probe_type}_emergence_layer_mse', None)
                    summary.append(f"  Finetuned Model (MSE):")
                    summary.append(f"    Best MSE: {fine_best_score_mse:.6f} at layer {fine_best_layer_mse}")
                    summary.append(f"    Emergence layer: {fine_em_mse if fine_em_mse is not None else 'None'}")
                    
                    base_em_mse = base_results.get(f'{probe_type}_emergence_layer_mse', None)
                    if base_em_mse is not None and fine_em_mse is not None:
                        shift = fine_em_mse - base_em_mse
                        summary.append(f"    Emergence shift: {shift:+d} layers")
            else:
                fine_scores = [fine_results[probe_key][l]['accuracy_test'] for l in range(32)]
                fine_best_layer = np.argmax(fine_scores)
                fine_best_score = fine_scores[fine_best_layer]
                fine_em = fine_results.get(f'{probe_type}_emergence_layer', None)
                summary.append(f"  Finetuned Model:")
                summary.append(f"    Best Accuracy: {fine_best_score:.3f} at layer {fine_best_layer}")
                summary.append(f"    Emergence layer: {fine_em if fine_em is not None else 'None'}")
                
                base_em = base_results.get(f'{probe_type}_emergence_layer', None)
                if base_em is not None and fine_em is not None:
                    shift = fine_em - base_em
                    summary.append(f"    Emergence shift: {shift:+d} layers")
                # Add class distribution if available
                class_dist = fine_results.get(f'{probe_type}_class_distribution', None)
                if class_dist is not None:
                    summary.append(f"    Class distribution (train): {class_dist}")
    
    summary.append("\n" + "="*80)
    
    summary_text = "\n".join(summary)
    with open(output_path / f"summary_{question_type}.txt", "w") as f:
        f.write(summary_text)
    
    print(f"  Saved: {output_path / f'summary_{question_type}.txt'}")
    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(
        description="Train layerwise probes to predict various targets from activations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
R² Interpretation:
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model performs as well as predicting the mean
  - R² < 0.0: Model performs WORSE than predicting the mean (very bad!)
  - R² = -200: Model's predictions are ~200x worse than just using the mean
  - Worst possible: -∞ (theoretically), but in practice limited by data variance

Severe Overfitting:
  When train R² is high (e.g., 0.99) but test R² is very negative (e.g., -200),
  the model memorized training data but fails on new data. This can be prevented by:
  - Increasing regularization (alpha parameter)
  - Using more training data
  - Using simpler models
  - Checking for data leakage between train/test splits
  - Verifying data quality (entropy values, scaling, etc.)
        """
    )
    parser.add_argument("--base_dir", type=str, required=True,
                       help="Base directory containing activation data (e.g., 'interp/activations/model_name')")
    parser.add_argument("--finetuned_dir", type=str, default=None,
                       help="Finetuned model directory (optional)")
    parser.add_argument("--output_dir", type=str, default="probe_results/improved",
                       help="Output directory for results")
    parser.add_argument("--question_type", type=str, nargs='+', default=['mcq'],
                       choices=['mcq', 'self', 'other'],
                       help="Question type(s) to analyze. Can specify multiple: --question_type mcq self other")
    parser.add_argument("--train_size", type=int, default=None,
                       help="Number of samples for training (rest for test). Default: 80%% split")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--regression_metric", type=str, default='r2',
                       choices=['r2', 'mse', 'both'],
                       help="Metric to use for regression probes (entropy/logit_margin). Options: 'r2', 'mse', or 'both'")
    parser.add_argument("--confidence_target", type=str, default='entropy',
                       choices=['entropy', 'logit_margin'],
                       help="Which confidence signal to use as the regression target: entropy or logit_margin.")
    parser.add_argument("--probe_types", type=str, nargs='+', default=None,
                       choices=['entropy', 'logit_margin', 'answer', 'correctness'],
                       help="Probe types to train. Can specify multiple: --probe_types entropy answer correctness. Default: all probes. Note: 'correctness' probe for self/other question types requires MCQ dataset to be loaded.")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Normalize question_type to list
    if isinstance(args.question_type, str):
        question_types = [args.question_type]
    else:
        question_types = args.question_type
    
    # Normalize probe_types to list
    if args.probe_types is None:
        probe_types = None  # Will use defaults
    elif isinstance(args.probe_types, str):
        probe_types = [args.probe_types]
    else:
        probe_types = args.probe_types
    
    print(f"\n{'='*80}")
    print("IMPROVED LAYERWISE PROBING")
    print(f"{'='*80}\n")
    print(f"Question types: {question_types}")
    print(f"Probe types: {probe_types if probe_types else 'all (default)'}")
    print(f"Base dir: {args.base_dir}")
    if args.finetuned_dir:
        print(f"Finetuned dir: {args.finetuned_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Regression metric: {args.regression_metric}")
    print(f"Confidence target: {args.confidence_target}\n")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each question type
    all_results = {}
    
    for question_type in question_types:
        print(f"\n{'='*80}")
        print(f"PROCESSING QUESTION TYPE: {question_type.upper()}")
        print(f"{'='*80}\n")
        
        # Map question_type to directory name
        if question_type == 'mcq':
            condition_dir = 'mcq'
        elif question_type == 'self':
            condition_dir = 'self_conf'
        elif question_type == 'other':
            condition_dir = 'other_conf'
        else:
            raise ValueError(f"Unknown question_type: {question_type}")
        
        # Load base model
        print(f"Loading base model data from {args.base_dir}/{condition_dir}...")
        try:
            base_dataset = ActivationDataset(f"{args.base_dir}/{condition_dir}")
        except Exception as e:
            print(f"⚠️  Could not load {condition_dir}: {e}")
            continue
        
        # Load MCQ dataset if needed for correctness probe (self/other)
        base_mcq_dataset = None
        if question_type in ['self', 'other']:
            print(f"Loading MCQ dataset for correctness lookup from {args.base_dir}/mcq...")
            try:
                base_mcq_dataset = ActivationDataset(f"{args.base_dir}/mcq")
                print(f"  ✓ Loaded MCQ dataset with {base_mcq_dataset.num_questions} questions")
            except Exception as e:
                print(f"  ⚠️  Could not load MCQ dataset: {e}")
                print(f"     Correctness probe will be skipped for {question_type}")
        
        # Create train/test split
        num_questions = base_dataset.num_questions
        if args.train_size is None:
            train_size = int(0.8 * num_questions)
        else:
            train_size = args.train_size
        
        all_indices = np.arange(num_questions)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:train_size].tolist()
        test_indices = all_indices[train_size:].tolist()
        
        print(f"Train/test split: {len(train_indices)}/{len(test_indices)}\n")
        
        # Train all probes for base model
        print("="*80)
        print("BASE MODEL")
        print("="*80)
        base_results = train_all_probes_for_condition(
            base_dataset, train_indices, test_indices, question_type, base_mcq_dataset,
            output_dir=args.output_dir, model_name='base', regression_metric=args.regression_metric,
            probe_types=probe_types, confidence_target=args.confidence_target
        )
        
        # Save plots (with error handling so failures don't lose results)
        print("\n" + "="*80)
        print("CREATING PLOTS (Base Model)")
        print("="*80)
        try:
            plot_all_probes(base_results, args.output_dir, 'base', regression_metric=args.regression_metric)
        except Exception as e:
            print(f"  ⚠️  Error plotting all probes: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            plot_confusion_matrices(base_results, args.output_dir, 'base', question_type)
        except Exception as e:
            print(f"  ⚠️  Error plotting confusion matrices: {e}")
            import traceback
            traceback.print_exc()
        
        # Train for finetuned if provided
        fine_results = None
        if args.finetuned_dir:
            print("\n" + "="*80)
            print("FINETUNED MODEL")
            print("="*80)
            try:
                fine_dataset = ActivationDataset(f"{args.finetuned_dir}/{condition_dir}")
            except Exception as e:
                print(f"⚠️  Could not load finetuned {condition_dir}: {e}")
                fine_dataset = None
            
            if fine_dataset:
                # Load MCQ dataset for finetuned if needed
                fine_mcq_dataset = None
                if question_type in ['self', 'other']:
                    try:
                        fine_mcq_dataset = ActivationDataset(f"{args.finetuned_dir}/mcq")
                    except Exception as e:
                        print(f"  ⚠️  Could not load finetuned MCQ dataset: {e}")
                        fine_mcq_dataset = None
                
                fine_results = train_all_probes_for_condition(
                    fine_dataset, train_indices, test_indices, question_type, fine_mcq_dataset,
                    output_dir=args.output_dir, model_name='finetuned', regression_metric=args.regression_metric,
                    probe_types=probe_types, confidence_target=args.confidence_target
                )
                
                # Save plots (with error handling)
                print("\n" + "="*80)
                print("CREATING PLOTS (Finetuned Model)")
                print("="*80)
                try:
                    plot_all_probes(fine_results, args.output_dir, 'finetuned', regression_metric=args.regression_metric)
                except Exception as e:
                    print(f"  ⚠️  Error plotting all probes: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    plot_confusion_matrices(fine_results, args.output_dir, 'finetuned', question_type)
                except Exception as e:
                    print(f"  ⚠️  Error plotting confusion matrices: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Comparison plots
                print("\n" + "="*80)
                print("CREATING COMPARISON PLOTS")
                print("="*80)
                try:
                    plot_comparison(base_results, fine_results, args.output_dir, question_type, regression_metric=args.regression_metric)
                except Exception as e:
                    print(f"  ⚠️  Error plotting comparison: {e}")
                    import traceback
                    traceback.print_exc()
                
                try:
                    plot_delta_improvement(base_results, fine_results, args.output_dir, question_type, regression_metric=args.regression_metric)
                except Exception as e:
                    print(f"  ⚠️  Error plotting delta improvement: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Store results
        all_results[question_type] = {
            'base': base_results,
        }
        if fine_results:
            all_results[question_type]['finetuned'] = fine_results
        
        # Final save (results already saved incrementally, but save combined version)
        print("\n" + "="*80)
        print("FINAL SAVE")
        print("="*80)
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            results_json = convert_to_json_serializable(all_results[question_type])
            
            with open(output_path / f"results_{question_type}.json", "w") as f:
                json.dump(results_json, f, indent=2)
            print(f"  ✓ Saved: {output_path / f'results_{question_type}.json'}")
        except Exception as e:
            print(f"  ⚠️  Error saving final results: {e}")
            import traceback
            traceback.print_exc()
        
        # Create summary
        try:
            create_summary(base_results, fine_results, args.output_dir, question_type, regression_metric=args.regression_metric)
        except Exception as e:
            print(f"  ⚠️  Error creating summary: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    for question_type in question_types:
        print(f"\n{question_type.upper()}:")
        print(f"  - results_{question_type}.json: Raw probe results")
        print(f"  - summary_{question_type}.txt: Key findings")
        print(f"  - base_{question_type}_all_probes.png: Base model plots")
        if args.finetuned_dir:
            print(f"  - finetuned_{question_type}_all_probes.png: Finetuned model plots")
            print(f"  - comparison_{question_type}_all_probes.png: Base vs Finetuned")
            print(f"  - delta_{question_type}_all_probes.png: Improvement analysis")


if __name__ == "__main__":
    main()
