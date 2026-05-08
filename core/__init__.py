"""
Core utilities shared across runners and notebooks.

Modules left in this package after the legacy purge:
  - model_utils    — model loading, device detection, chat-template detection
  - datasets       — registered dataset loaders (TriviaMC, SimpleMC, GPQA, …)
  - metrics        — uncertainty signal computations (entropy, margin, logit_gap, …)
  - probes         — `LinearProbe` + transfer/permutation analyses (used by
                     analysis/analyze_activations.ipynb)
  - questions      — question-set hashing + cross-experiment consistency checks
  - logres_helpers — partial correlation / decision-on-controls helper

The earlier `extraction`, `directions`, `logit_lens`, `steering`,
`steering_experiments` modules were duplicates of (or now superseded by)
helpers in `interp_experiments/`. Direction extraction + logit lens for the
introspection pipeline live in `interp_experiments/_probes.py` and
`interp_experiments/_logit_lens.py`. Steering / ablation will live in
`interp_experiments/run_causal_interventions.py` once we build it.
"""

from .model_utils import (
    DEVICE,
    is_base_model,
    has_chat_template,
    get_model_short_name,
    get_run_name,
    load_model_and_tokenizer,
    should_use_chat_template,
)

from .metrics import (
    compute_entropy,
    compute_metrics_single,
    compute_mc_metrics,
    compute_nexttoken_metrics,
    METRIC_INFO,
    metric_sign_for_confidence,
)

from .probes import (
    LinearProbe,
    train_and_evaluate_probe,
    test_transfer,
    permutation_test,
    run_layer_analysis,
    compute_introspection_scores,
    train_introspection_mapping_probe,
    compute_contrastive_direction,
    extract_probe_direction,
    run_introspection_mapping_analysis,
)

from .questions import (
    load_questions,
    get_question_hash,
    save_question_set,
    load_question_set,
    verify_question_consistency,
    format_direct_prompt,
    split_questions,
)

__all__ = [
    # model_utils
    "DEVICE",
    "is_base_model",
    "has_chat_template",
    "get_model_short_name",
    "get_run_name",
    "load_model_and_tokenizer",
    "should_use_chat_template",
    # metrics
    "compute_entropy",
    "compute_metrics_single",
    "compute_mc_metrics",
    "compute_nexttoken_metrics",
    "METRIC_INFO",
    "metric_sign_for_confidence",
    # probes
    "LinearProbe",
    "train_and_evaluate_probe",
    "test_transfer",
    "permutation_test",
    "run_layer_analysis",
    "compute_introspection_scores",
    "train_introspection_mapping_probe",
    "compute_contrastive_direction",
    "extract_probe_direction",
    "run_introspection_mapping_analysis",
    # questions
    "load_questions",
    "get_question_hash",
    "save_question_set",
    "load_question_set",
    "verify_question_consistency",
    "format_direct_prompt",
    "split_questions",
]
