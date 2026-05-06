"""
Run vanilla logit lens analysis on a language model.

This script applies the logit lens technique to visualize how token predictions
evolve across layers of a transformer. At each layer, we project the residual
stream through the unembedding matrix to see what tokens the model would predict
if decoding stopped at that layer.

Usage examples:
    # Basic usage with default prompt (analyzes ALL layers)
    python run_logit_lens.py
    
    # Custom prompt (ignores TASK_TYPE, uses raw string)
    python run_logit_lens.py --prompt "The capital of France is"
    
    # Analyze specific layers
    python run_logit_lens.py --layers 0,20,40,79
    
    # Save output to JSON
    python run_logit_lens.py --out outputs/logit_lens_results.json
    
    # Use raw residuals (no LayerNorm)
    python run_logit_lens.py --ln-mode none

Configuration:
    Most settings can be changed via module-level constants below or CLI args.
    The constants are used as defaults when CLI args are not provided.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import argparse
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
)
from core.logit_lens import (
    LogitLensConfig,
    LogitLensAnalyzer,
    auto_select_layers,
    print_logit_lens_summary,
)
from prompts import (
    format_direct_prompt,
    format_stated_confidence_prompt,
    format_other_confidence_prompt,
    format_answer_or_delegate_prompt,
    STATED_CONFIDENCE_OPTIONS,
    OTHER_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
)

# =============================================================================
# Configuration — edit values in experiment_config.LogitLensConfig
# =============================================================================
from experiment_config import LogitLensConfig as _C

BASE_MODEL_NAME = _C.BASE_MODEL_NAME
ADAPTER = _C.ADAPTER

def get_output_prefix():
    if ADAPTER:
        return "Llama_3.1_8b_FT_"
    else:
        return "Llama_3.1_8b_instruct_"

USE_DATASET_QUESTION = _C.USE_DATASET_QUESTION
DATA_FILE = _C.DATA_FILE
QID = _C.QID
TEST_QUESTION = dict(_C.TEST_QUESTION)
TASK_TYPE = _C.TASK_TYPE
DEFAULT_PROMPT = _C.DEFAULT_PROMPT
ACTIVATION_STREAM = _C.ACTIVATION_STREAM
LN_MODE = _C.LN_MODE
TOKEN_POSITION = _C.TOKEN_POSITION
TOP_K = _C.TOP_K
LAYERS_DEFAULT = _C.LAYERS_DEFAULT
LOAD_IN_4BIT = _C.LOAD_IN_4BIT
LOAD_IN_8BIT = _C.LOAD_IN_8BIT
OUTPUTS_DIR = _C.OUTPUTS_DIR

# =============================================================================
# Main
# =============================================================================

def parse_layers(layers_str: str, num_layers: int) -> List[int]:
    """
    Parse layers argument.
    
    Args:
        layers_str: "all", "auto", or comma-separated layer indices (e.g., "0,20,40")
        num_layers: Total layers in model
    
    Returns:
        List of layer indices
    """
    if layers_str.lower() == "all":
        return list(range(num_layers))
    
    if layers_str.lower() == "auto":
        return auto_select_layers(num_layers)
    
    try:
        layers = [int(x.strip()) for x in layers_str.split(",")]
        # Validate
        for layer in layers:
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Layer {layer} out of range [0, {num_layers-1}]")
        return sorted(set(layers))
    except ValueError as e:
        raise ValueError(f"Invalid layers format '{layers_str}': {e}")


def parse_token_position(pos_str: str):
    """Parse token position argument."""
    if pos_str.lower() == "last":
        return "last"
    try:
        return int(pos_str)
    except ValueError:
        raise ValueError(f"token_position must be 'last' or an integer, got '{pos_str}'")


def format_prompt_for_task(
    task_type: str,
    question: Dict[str, Any],
    tokenizer,
    trial_index: int = 0,
) -> tuple[str, List[str], str]:
    """
    Format a prompt according to the specified task type.
    
    Args:
        task_type: One of "direct_mc", "stated_confidence", "other_confidence", 
                   "answer_or_delegate", or "raw"
        question: Question dict with 'question' and 'options' keys
        tokenizer: Tokenizer for chat template formatting
        trial_index: For answer_or_delegate, controls which digit means answer/delegate
    
    Returns:
        Tuple of (formatted_prompt, option_tokens, task_description)
        - formatted_prompt: The full prompt string with chat template applied
        - option_tokens: List of valid response tokens (e.g., ["A", "B", "C", "D"])
        - task_description: Human-readable description of the task
    """
    if task_type == "direct_mc":
        prompt, options = format_direct_prompt(question, tokenizer, use_chat_template=True)
        return prompt, options, "Direct MC (A/B/C/D)"
    
    elif task_type == "stated_confidence":
        prompt, options = format_stated_confidence_prompt(question, tokenizer, use_chat_template=True)
        return prompt, options, "Stated Confidence (S-Z)"
    
    elif task_type == "other_confidence":
        prompt, options = format_other_confidence_prompt(question, tokenizer, use_chat_template=True)
        return prompt, options, "Other Confidence (S-Z)"
    
    elif task_type == "answer_or_delegate":
        prompt, options, mapping = format_answer_or_delegate_prompt(
            question, tokenizer, trial_index=trial_index, use_chat_template=True
        )
        # Include mapping info in description
        desc = f"Answer/Delegate (1={mapping['1']}, 2={mapping['2']})"
        return prompt, options, desc
    
    elif task_type == "raw":
        # No formatting, return DEFAULT_PROMPT as-is
        return DEFAULT_PROMPT, [], "Raw prompt (no task formatting)"
    
    else:
        raise ValueError(
            f"Unknown task_type: {task_type}. "
            f"Valid options: direct_mc, stated_confidence, other_confidence, answer_or_delegate, raw"
        )


def load_question_from_dataset(data_file: str, qid: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a question from a JSONL dataset file.
    
    Args:
        data_file: Path to JSONL file with questions
        qid: Specific question ID to load, or None for random selection
    
    Returns:
        Question dict formatted for use with format_prompt_for_task
        (has 'question', 'options' dict with A/B/C/D keys, and 'correct_answer')
    """
    questions = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    if not questions:
        raise ValueError(f"No questions found in {data_file}")
    
    # Select question
    if qid is not None:
        # Find by qid
        question_data = None
        for q in questions:
            if q.get('qid') == qid:
                question_data = q
                break
        if question_data is None:
            raise ValueError(f"Question with qid '{qid}' not found in {data_file}")
    else:
        # Random selection
        question_data = random.choice(questions)
    
    # Convert to format expected by format_prompt_for_task
    # TriviaMC format: {"qid": "...", "question": "...", "correct_answer": "...", "distractors": [...]}
    # Need to convert to: {"question": "...", "options": {"A": ..., "B": ..., "C": ..., "D": ...}, "correct_answer": "A/B/C/D"}
    
    # Combine correct answer and distractors, shuffle them
    all_options = [question_data['correct_answer']] + question_data['distractors']
    random.shuffle(all_options)
    
    # Create options dict and find which letter is correct
    option_letters = ['A', 'B', 'C', 'D']
    options = {}
    correct_letter = None
    
    for i, opt in enumerate(all_options):
        letter = option_letters[i]
        options[letter] = opt
        if opt == question_data['correct_answer']:
            correct_letter = letter
    
    return {
        'qid': question_data['qid'],
        'question': question_data['question'],
        'options': options,
        'correct_answer': correct_letter,
    }


def compute_entropy(logits: np.ndarray) -> float:
    """Compute entropy of softmax distribution from logits."""
    # Shift for numerical stability
    logits = logits - logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    # Avoid log(0) by filtering zeros
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def generate_txt_report(
    report: Dict[str, Any],
    args,
    analyzer: LogitLensAnalyzer,
) -> str:
    """
    Generate a comprehensive text report of logit lens results.
    
    Args:
        report: Output from LogitLensAnalyzer.logit_lens_report
        args: Command-line arguments
        analyzer: The LogitLensAnalyzer (for accessing cached logits)
    
    Returns:
        String containing the full text report
    """
    lines = []
    
    # Header
    lines.append("=" * 100)
    lines.append("LOGIT LENS ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    
    # Parameters section
    lines.append("-" * 50)
    lines.append("PARAMETERS")
    lines.append("-" * 50)
    lines.append(f"Timestamp:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model:              {report['metadata']['model']}")
    lines.append(f"Model (short):      {report['metadata']['model_short']}")
    lines.append(f"Total layers:       {report['metadata']['num_layers']}")
    lines.append(f"Device:             {report['metadata']['device']}")
    
    # Task information
    if 'task_type' in report['metadata']:
        lines.append(f"Task type:          {report['metadata']['task_type']}")
        lines.append(f"Task description:   {report['metadata']['task_description']}")
    if 'option_tokens' in report['metadata'] and report['metadata']['option_tokens']:
        lines.append(f"Option tokens:      {report['metadata']['option_tokens']}")
    
    # Question info if present
    if 'question' in report:
        lines.append("")
        lines.append("Question:")
        lines.append(f"  {report['question']['question']}")
        if 'options' in report['question']:
            for k, v in report['question']['options'].items():
                lines.append(f"    {k}: {v}")
        if 'correct_answer' in report['question']:
            lines.append(f"  Correct answer: {report['question']['correct_answer']}")
    
    lines.append("")
    lines.append(f"Input tokens:       {report['num_tokens']}")
    lines.append(f"Token position:     {report['config']['token_position']}")
    lines.append(f"Activation stream:  {report['config']['activation_stream']}")
    lines.append(f"LayerNorm mode:     {report['config']['ln_mode']}")
    lines.append(f"Top K:              {report['config']['top_k']}")
    lines.append(f"Layers analyzed:    {len(report['layers_analyzed'])} layers ({report['layers_analyzed'][0]} to {report['layers_analyzed'][-1]})")
    lines.append("")
    
    # Full prompt (useful for seeing exact chat template formatting)
    lines.append("-" * 50)
    lines.append("FULL PROMPT (as sent to model)")
    lines.append("-" * 50)
    lines.append(report['prompt'])
    lines.append("")
    
    # Tokenization
    lines.append("-" * 50)
    lines.append("TOKENIZATION")
    lines.append("-" * 50)
    for i, tok in enumerate(report['input_tokens']):
        lines.append(f"  [{i:3d}] {tok}")
    lines.append("")
    
    # Final output (ground truth for comparison)
    lines.append("-" * 50)
    lines.append("FINAL MODEL OUTPUT (Reference)")
    lines.append("-" * 50)
    final_top = report['final_output']['top_tokens']['top_positive']
    lines.append(f"Top-1 prediction: {final_top[0]['token_str']} (logit={final_top[0]['logit']:.4f})")
    lines.append("")
    
    # Summary table across layers
    lines.append("-" * 50)
    lines.append("LAYER SUMMARY (tracking final answer)")
    lines.append("-" * 50)
    
    final_answer_token_id = final_top[0]['token_id']
    final_answer_token_str = final_top[0]['token_str']
    
    lines.append(f"Final answer token: {final_answer_token_str} (id={final_answer_token_id})")
    lines.append("")
    lines.append(f"{'Layer':>6} | {'Top-1 Prediction':<25} | {'Top-1 Logit':>11} | {'Top-1 Prob':>10} | {'Entropy':>8} | {'Final Ans Rank':>14} | {'Final Ans Logit':>15}")
    lines.append("-" * 120)
    
    for layer_str in sorted(report['layer_results'].keys(), key=int):
        layer_data = report['layer_results'][layer_str]
        top_tokens = layer_data['top_tokens']['top_positive']
        
        # Top-1 prediction
        top1 = top_tokens[0]
        top1_str = top1['token_str'][:22] + "..." if len(top1['token_str']) > 25 else top1['token_str']
        top1_logit = top1['logit']
        
        # Compute probability and entropy from all logits
        # We need to reconstruct the full logit vector - we stored stats but not full logits in the report
        # For now, estimate entropy from the logits range/std
        # Actually, let's compute proper entropy from the stored data if available
        logits_std = layer_data['logits_std']
        logits_max = layer_data['logits_max']
        logits_mean = layer_data['logits_mean']
        
        # Estimate top-1 probability using softmax approximation
        # This is rough - the exact prob requires all logits
        # Let's use: prob ≈ exp(max - mean - std^2/2) / vocab_size approximately
        # Better: use logits_max - second highest (we have that in top_positive[1])
        if len(top_tokens) > 1:
            gap = top1_logit - top_tokens[1]['logit']
            # Prob ≈ sigmoid(gap) for 2-way, but for full vocab it's lower
            # Rough estimate: if gap is large (>5), prob is high
            # This is a rough proxy since we don't have full logits
            # Let's just show the gap instead
            top1_prob_approx = f"gap={gap:.2f}"
        else:
            top1_prob_approx = "N/A"
        
        # Find rank of final answer token in this layer
        final_ans_rank = "N/A"
        final_ans_logit = "N/A"
        for rank, tok in enumerate(top_tokens):
            if tok['token_id'] == final_answer_token_id:
                final_ans_rank = str(rank + 1)
                final_ans_logit = f"{tok['logit']:.4f}"
                break
        else:
            # Not in top-k, check bottom
            bottom_tokens = layer_data['top_tokens']['top_negative']
            for tok in bottom_tokens:
                if tok['token_id'] == final_answer_token_id:
                    final_ans_rank = f">{len(top_tokens)}"
                    final_ans_logit = f"{tok['logit']:.4f}"
                    break
            else:
                final_ans_rank = f">{len(top_tokens)}"
        
        # Entropy estimate from std (Shannon entropy of Gaussian ≈ 0.5*ln(2πe*σ^2))
        entropy_est = f"{logits_std:.2f}σ"
        
        lines.append(f"{int(layer_str):>6} | {top1_str:<25} | {top1_logit:>11.4f} | {top1_prob_approx:>10} | {entropy_est:>8} | {final_ans_rank:>14} | {final_ans_logit:>15}")
    
    # If we have option tokens, show a table tracking each option across layers
    if 'option_tokens' in report.get('metadata', {}) and report['metadata']['option_tokens']:
        option_tokens = report['metadata']['option_tokens']
        lines.append("")
        lines.append("-" * 50)
        lines.append("OPTION TOKEN TRACKING ACROSS LAYERS")
        lines.append("-" * 50)
        lines.append(f"Tracking logits for option tokens: {option_tokens}")
        lines.append("")
        
        # Build header
        header = f"{'Layer':>6}"
        for opt in option_tokens:
            header += f" | {opt:>10}"
        header += " | Winner"
        lines.append(header)
        lines.append("-" * (10 + 13 * len(option_tokens) + 10))
        
        for layer_str in sorted(report['layer_results'].keys(), key=int):
            layer_data = report['layer_results'][layer_str]
            top_tokens = layer_data['top_tokens']['top_positive']
            bottom_tokens = layer_data['top_tokens']['top_negative']
            
            # Find logits for each option token
            row = f"{int(layer_str):>6}"
            option_logits = {}
            
            for opt in option_tokens:
                opt_logit = None
                # Search in top tokens
                for tok in top_tokens:
                    # Token strings are repr'd, so check both with and without repr
                    if tok['token_str'] == repr(opt) or tok['token_str'] == f"'{opt}'":
                        opt_logit = tok['logit']
                        break
                # Search in bottom tokens if not found
                if opt_logit is None:
                    for tok in bottom_tokens:
                        if tok['token_str'] == repr(opt) or tok['token_str'] == f"'{opt}'":
                            opt_logit = tok['logit']
                            break
                
                if opt_logit is not None:
                    row += f" | {opt_logit:>10.4f}"
                    option_logits[opt] = opt_logit
                else:
                    row += f" | {'N/A':>10}"
            
            # Determine winner among options
            if option_logits:
                winner = max(option_logits.keys(), key=lambda k: option_logits[k])
                row += f" | {winner:>6}"
            else:
                row += f" | {'?':>6}"
            
            lines.append(row)
    
    lines.append("")
    
    # Detailed layer-by-layer results
    lines.append("=" * 100)
    lines.append("DETAILED LAYER-BY-LAYER RESULTS")
    lines.append("=" * 100)
    
    for layer_str in sorted(report['layer_results'].keys(), key=int):
        layer_data = report['layer_results'][layer_str]
        
        lines.append("")
        lines.append(f"{'=' * 40} LAYER {layer_str} {'=' * 40}")
        lines.append(f"Logits stats: mean={layer_data['logits_mean']:.4f}, std={layer_data['logits_std']:.4f}, "
                    f"min={layer_data['logits_min']:.4f}, max={layer_data['logits_max']:.4f}")
        lines.append("")
        
        # Top 20 positive
        lines.append(f"TOP {len(layer_data['top_tokens']['top_positive'])} TOKENS (highest logits):")
        lines.append(f"{'Rank':>4} | {'Token':<30} | {'Token ID':>10} | {'Logit':>12}")
        lines.append("-" * 65)
        for rank, tok in enumerate(layer_data['top_tokens']['top_positive']):
            tok_str = tok['token_str'][:27] + "..." if len(tok['token_str']) > 30 else tok['token_str']
            lines.append(f"{rank+1:>4} | {tok_str:<30} | {tok['token_id']:>10} | {tok['logit']:>12.4f}")
        
        lines.append("")
        
        # Bottom 20 negative  
        lines.append(f"BOTTOM {len(layer_data['top_tokens']['top_negative'])} TOKENS (lowest logits):")
        lines.append(f"{'Rank':>4} | {'Token':<30} | {'Token ID':>10} | {'Logit':>12}")
        lines.append("-" * 65)
        for rank, tok in enumerate(layer_data['top_tokens']['top_negative']):
            tok_str = tok['token_str'][:27] + "..." if len(tok['token_str']) > 30 else tok['token_str']
            lines.append(f"{rank+1:>4} | {tok_str:<30} | {tok['token_id']:>10} | {tok['logit']:>12.4f}")
    
    lines.append("")
    lines.append("=" * 100)
    lines.append("END OF REPORT")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def compute_token_probabilities_across_layers(
    analyzer: LogitLensAnalyzer,
    prompt: str,
    layers: List[int],
    token_ids: List[int],
    config: LogitLensConfig,
) -> Dict[int, np.ndarray]:
    """
    Compute probabilities for specific tokens across all layers.
    
    Args:
        analyzer: LogitLensAnalyzer instance
        prompt: Input prompt
        layers: List of layer indices to analyze
        token_ids: Token IDs to track
        config: LogitLensConfig
    
    Returns:
        Dict mapping token_id -> np.array of probabilities across layers
    """
    # Run forward pass and cache activations
    tokens, final_logits, cache = analyzer.run_and_cache(
        prompt, layers, config.activation_stream
    )
    
    # Store probabilities for each tracked token at each layer
    token_probs = {tid: [] for tid in token_ids}
    
    for layer in layers:
        # Get residual activations at this layer
        residuals = analyzer.get_residual_activations(cache, layer)
        
        # Select token position
        if config.token_position == "last":
            token_vec = residuals[-1]
        else:
            token_vec = residuals[config.token_position]
        
        # Compute logits via unembedding
        from core.logit_lens import unembed_vector_to_logits
        layer_logits = unembed_vector_to_logits(
            analyzer.model,
            token_vec,
            ln_mode=config.ln_mode,
            W_U=analyzer.W_U,
            b_U=analyzer.b_U,
            final_ln=analyzer.final_ln
        )
        
        # Convert to probabilities via softmax
        probs = F.softmax(layer_logits, dim=0).detach().cpu().numpy()
        
        # Store probabilities for tracked tokens
        for tid in token_ids:
            token_probs[tid].append(probs[tid])
    
    # Convert to numpy arrays
    return {tid: np.array(probs_list) for tid, probs_list in token_probs.items()}


def get_top1_tokens_per_layer(
    analyzer: LogitLensAnalyzer,
    prompt: str,
    layers: List[int],
    config: LogitLensConfig,
) -> Dict[int, int]:
    """
    Get the top-1 token for each layer.
    
    Args:
        analyzer: LogitLensAnalyzer instance
        prompt: Input prompt
        layers: List of layer indices
        config: LogitLensConfig
    
    Returns:
        Dict mapping layer -> top-1 token ID at that layer
    """
    # Run forward pass and cache activations
    tokens, final_logits, cache = analyzer.run_and_cache(
        prompt, layers, config.activation_stream
    )
    
    layer_top1 = {}
    
    for layer in layers:
        # Get residual activations at this layer
        residuals = analyzer.get_residual_activations(cache, layer)
        
        # Select token position
        if config.token_position == "last":
            token_vec = residuals[-1]
        else:
            token_vec = residuals[config.token_position]
        
        # Compute logits via unembedding
        from core.logit_lens import unembed_vector_to_logits
        layer_logits = unembed_vector_to_logits(
            analyzer.model,
            token_vec,
            ln_mode=config.ln_mode,
            W_U=analyzer.W_U,
            b_U=analyzer.b_U,
            final_ln=analyzer.final_ln
        )
        
        # Get top-1 token
        top1_idx = layer_logits.argmax().item()
        layer_top1[layer] = top1_idx
    
    return layer_top1


def plot_top5_output_tokens_across_layers(
    analyzer: LogitLensAnalyzer,
    report: Dict[str, Any],
    prompt: str,
    layers: List[int],
    config: LogitLensConfig,
    output_path: Path,
    title_suffix: str = "",
    peak_label_threshold: float = 0.2,
):
    """
    Plot probability of top 5 output tokens across all layers.
    
    Takes the top 5 tokens from the final output layer and tracks their
    probability trajectory across all intermediate layers.
    
    Args:
        analyzer: LogitLensAnalyzer instance
        report: Output from LogitLensAnalyzer.logit_lens_report
        prompt: Input prompt
        layers: List of layer indices
        config: LogitLensConfig
        output_path: Where to save the plot
        title_suffix: Additional text to append to the plot title (e.g., dataset and qid)
        peak_label_threshold: Add label at peak for tokens exceeding this probability (default: 0.2)
    """
    # Get top 5 token IDs from final output
    final_top5 = report["final_output"]["top_tokens"]["top_positive"][:5]
    top5_token_ids = [tok["token_id"] for tok in final_top5]
    top5_token_strs = [tok["token_str"] for tok in final_top5]
    
    # Compute probabilities for these tokens across all layers
    token_probs = compute_token_probabilities_across_layers(
        analyzer, prompt, layers, top5_token_ids, config
    )
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10.colors[:5]
    
    for i, (tid, token_str) in enumerate(zip(top5_token_ids, top5_token_strs)):
        probs = token_probs[tid]
        # Clean up token string for legend (remove repr quotes)
        clean_label = token_str.strip("'\"")
        ax.plot(layers, probs, 'o-', color=colors[i], 
                label=f"{clean_label}", markersize=4, linewidth=2)
        
        # Add peak label if max probability exceeds threshold
        max_prob = np.max(probs)
        if max_prob >= peak_label_threshold:
            peak_idx = np.argmax(probs)
            peak_layer = layers[peak_idx]
            peak_prob = probs[peak_idx]
            
            ax.annotate(
                clean_label,
                xy=(peak_layer, peak_prob),
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color=colors[i],
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=colors[i], alpha=0.8),
            )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    title = f'Top 5 Output Tokens: Probability Across Layers{title_suffix}\n(labels at peaks > 0.2)'
    ax.set_title(title, fontsize=14)
    ax.legend(title='Token', loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_top1_tokens_across_layers(
    analyzer: LogitLensAnalyzer,
    prompt: str,
    layers: List[int],
    config: LogitLensConfig,
    output_path: Path,
    title_suffix: str = "",
    peak_label_threshold: float = 0.2,
):
    """
    Plot how each layer's top-1 token rises and falls across all layers.
    
    For each layer, identifies its top-1 token, then tracks how that token's
    probability evolves across all layers. Creates a line chart with one line
    per layer's top token.
    
    Args:
        analyzer: LogitLensAnalyzer instance
        prompt: Input prompt
        layers: List of layer indices
        config: LogitLensConfig
        output_path: Where to save the plot
        peak_label_threshold: Add label at peak for tokens exceeding this probability (default: 0.2)
    """
    # Get top-1 token for each layer
    layer_top1 = get_top1_tokens_per_layer(analyzer, prompt, layers, config)
    
    # Get unique tokens to track (some layers may share the same top-1)
    unique_tokens = list(set(layer_top1.values()))
    
    # Compute probabilities for all unique tokens across all layers
    token_probs = compute_token_probabilities_across_layers(
        analyzer, prompt, layers, unique_tokens, config
    )
    
    # Create figure with enough space for legend
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate distinct colors for each layer
    num_layers = len(layers)
    # Use a colormap that provides good distinction
    cmap = plt.cm.get_cmap('turbo', num_layers)
    
    # Track which tokens we've already labeled to avoid duplicate legend entries
    labeled_tokens = set()
    lines_info = []  # Store (line, layer, token_str, final_prob, probs, color, tid) for legend sorting
    
    for i, layer in enumerate(layers):
        top1_tid = layer_top1[layer]
        probs = token_probs[top1_tid]
        
        # Get token string for label
        token_str = analyzer.tokenizer.decode([top1_tid])
        clean_str = repr(token_str).strip("'\"")
        
        color = cmap(i / (num_layers - 1) if num_layers > 1 else 0)
        
        # Create label with layer number and token
        label = f"L{layer}: {clean_str}"
        
        line, = ax.plot(layers, probs, '-', color=color, 
                       label=label, linewidth=1.5, alpha=0.8)
        
        # Mark the layer where this is the top-1 with a marker
        ax.plot(layer, probs[layers.index(layer)], 'o', color=color, 
               markersize=6, markeredgecolor='black', markeredgewidth=0.5)
        
        lines_info.append((line, layer, clean_str, probs[-1], probs, color, top1_tid))
    
    # Add peak labels for tokens that exceed the threshold
    # Track labeled positions to avoid overlapping labels
    labeled_peaks = set()  # Set of (token_id) to avoid labeling same token twice
    
    for line, layer, clean_str, final_prob, probs, color, tid in lines_info:
        max_prob = np.max(probs)
        
        # Only label if max prob exceeds threshold and we haven't labeled this token yet
        if max_prob >= peak_label_threshold and tid not in labeled_peaks:
            labeled_peaks.add(tid)
            
            # Find the peak layer index
            peak_idx = np.argmax(probs)
            peak_layer = layers[peak_idx]
            peak_prob = probs[peak_idx]
            
            # Add annotation at the peak
            ax.annotate(
                clean_str,
                xy=(peak_layer, peak_prob),
                xytext=(0, 8),  # Offset label slightly above the point
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color=color,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, alpha=0.8),
            )
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    title = f'Layer Top-1 Tokens: Probability Trajectories{title_suffix}\n(marker indicates layer where token is top-1, labels at peaks > 0.2)'
    ax.set_title(title, fontsize=14)
    
    # Create legend with smaller font and outside the plot
    # Sort by final probability (descending) for easier reading
    lines_info.sort(key=lambda x: -x[3])
    
    # Create a legend outside the plot on the right side
    handles = [info[0] for info in lines_info]
    labels = [f"L{info[1]}: {info[2]}" for info in lines_info]
    
    # Split into two columns if many layers
    ncol = 2 if num_layers > 16 else 1
    legend = ax.legend(handles, labels, 
                      loc='center left', 
                      bbox_to_anchor=(1.02, 0.5),
                      fontsize=8,
                      ncol=ncol,
                      title='Layer: Token')
    legend.get_title().set_fontsize(9)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run logit lens analysis on a language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--model", type=str, default=BASE_MODEL_NAME,
        help=f"Model name or path (default: {BASE_MODEL_NAME})"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Raw prompt to analyze (overrides TASK_TYPE formatting)"
    )
    parser.add_argument(
        "--task", type=str, default=TASK_TYPE,
        choices=["direct_mc", "stated_confidence", "other_confidence", "answer_or_delegate", "raw"],
        help=f"Task type for formatting TEST_QUESTION (default: {TASK_TYPE})"
    )
    parser.add_argument(
        "--layers", type=str, default=LAYERS_DEFAULT,
        help="'all' for every layer, 'auto' for [0, mid, last], or comma-separated indices (default: all)"
    )
    parser.add_argument(
        "--activation-stream", type=str, default=ACTIVATION_STREAM,
        choices=["resid_pre", "resid_post"],
        help=f"Which activation stream to read (default: {ACTIVATION_STREAM})"
    )
    parser.add_argument(
        "--token-position", type=str, default=str(TOKEN_POSITION),
        help=f"Token position to analyze: 'last' or integer (default: {TOKEN_POSITION})"
    )
    parser.add_argument(
        "--ln-mode", type=str, default=LN_MODE,
        choices=["none", "final_ln", "model_default"],
        help=f"LayerNorm mode (default: {LN_MODE})"
    )
    parser.add_argument(
        "--k", type=int, default=TOP_K,
        help=f"Number of top tokens to report (default: {TOP_K})"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Path to save JSON output (default: auto-generated in outputs/)"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", default=LOAD_IN_4BIT,
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", default=LOAD_IN_8BIT,
        help="Load model in 8-bit quantization"
    )
    
    args = parser.parse_args()
    
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    
    # Load model (need tokenizer for prompt formatting)
    model, tokenizer, num_layers = load_model_and_tokenizer(
        args.model,
        adapter_path=ADAPTER,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    
    # Determine the prompt to use
    if args.prompt is not None:
        # User provided explicit prompt - use it directly
        prompt = args.prompt
        task_desc = "Raw prompt (--prompt override)"
        option_tokens = []
        question_data = None
        print(f"Task: {task_desc}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    else:
        # Select question source
        if USE_DATASET_QUESTION:
            # Load question from dataset
            question_data = load_question_from_dataset(DATA_FILE, QID)
            print(f"Loaded question from {DATA_FILE}: {question_data['qid']}")
        else:
            # Use hardcoded TEST_QUESTION
            question_data = TEST_QUESTION
        
        # Format question according to task type
        prompt, option_tokens, task_desc = format_prompt_for_task(
            args.task, question_data, tokenizer
        )
        print(f"Task: {task_desc}")
        print(f"Question: {question_data['question']}")
        print(f"Options: {question_data['options']}")
        print(f"Correct answer: {question_data['correct_answer']}")
        if option_tokens:
            print(f"Expected response tokens: {option_tokens}")
    print()
    
    # Parse arguments
    layers = parse_layers(args.layers, num_layers)
    token_position = parse_token_position(args.token_position)
    
    print(f"Analyzing {len(layers)} layers: [{layers[0]}..{layers[-1]}]" if len(layers) > 5 else f"Analyzing layers: {layers}")
    print(f"Token position: {token_position}")
    print(f"Activation stream: {args.activation_stream}")
    print(f"LN mode: {args.ln_mode}")
    print()
    
    # Create config
    config = LogitLensConfig(
        activation_stream=args.activation_stream,
        ln_mode=args.ln_mode,
        token_position=token_position,
        top_k=args.k,
    )
    
    # Run analysis
    analyzer = LogitLensAnalyzer(model, tokenizer, num_layers)
    report = analyzer.logit_lens_report(prompt, layers, config)
    
    # Add metadata
    report["metadata"] = {
        "model": args.model,
        "model_short": get_model_short_name(args.model),
        "adapter": ADAPTER,
        "num_layers": num_layers,
        "device": str(DEVICE),
        "task_type": args.task if args.prompt is None else "raw",
        "task_description": task_desc,
        "option_tokens": option_tokens,
    }
    
    # Add question info if using task formatting
    if args.prompt is None and question_data is not None:
        report["question"] = question_data
    
    # Print summary (brief version to console)
    print_logit_lens_summary(report, top_n=10)
    
    # Save outputs
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    # Use the standardized prefix based on adapter
    output_prefix = get_output_prefix()
    
    # Build filename suffix with task type, dataset, and qid if available
    task_type = args.task if args.prompt is None else "raw"
    if question_data is not None and 'qid' in question_data:
        dataset_name = Path(DATA_FILE).stem  # e.g., "TriviaMC"
        qid_number = question_data['qid'].split("_")[-1]  # e.g., "146" from "triviamc_146"
        suffix = f"_{task_type}_{dataset_name}_{qid_number}"
    else:
        suffix = f"_{task_type}"
    
    # JSON output path
    if args.out:
        json_path = Path(args.out)
        txt_path = json_path.with_suffix(".txt")
        base_name = json_path.stem
    else:
        json_path = OUTPUTS_DIR / f"{output_prefix}logit_lens{suffix}.json"
        txt_path = OUTPUTS_DIR / f"{output_prefix}logit_lens{suffix}.txt"
        base_name = f"{output_prefix}logit_lens{suffix}"
    
    # Save JSON
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON results saved to {json_path}")
    
    # Generate and save comprehensive text report
    txt_report = generate_txt_report(report, args, analyzer)
    with open(txt_path, "w") as f:
        f.write(txt_report)
    print(f"Text report saved to {txt_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Build title suffix for plots (e.g., " - FT - stated_confidence - TriviaMC #146")
    model_type = "FT" if ADAPTER else "Instruct"
    if question_data is not None and 'qid' in question_data:
        title_suffix = f" - {model_type} - {task_type} - {dataset_name} #{qid_number}"
    else:
        title_suffix = f" - {model_type} - {task_type}"
    
    # Plot 1: Top 5 output tokens tracked across layers
    top5_plot_path = OUTPUTS_DIR / f"{output_prefix}logit_lens_top5_tokens{suffix}.png"
    plot_top5_output_tokens_across_layers(
        analyzer, report, prompt, layers, config, top5_plot_path,
        title_suffix=title_suffix
    )
    
    # Plot 2: Each layer's top-1 token tracked across all layers
    layer_top1_plot_path = OUTPUTS_DIR / f"{output_prefix}logit_lens_layer_top1_tokens{suffix}.png"
    plot_layer_top1_tokens_across_layers(
        analyzer, prompt, layers, config, layer_top1_plot_path,
        title_suffix=title_suffix
    )


if __name__ == "__main__":
    main()
