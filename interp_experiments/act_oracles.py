"""
Interpret linear probe directions using Activation Oracles.

This script loads direction vectors from run_contrastive_direction.py outputs
and uses the Activation Oracle (AO) adapter to interpret what concepts they represent.

Usage:
    python act_oracles.py
"""

# --- repo path bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from preflight import check_hf_login; check_hf_login()



import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Lazy imports for heavy dependencies (torch, peft)
# Only imported when running full interpretation, not for analysis-only mode

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path if using fine-tuned model
DATASET_NAME = "TriviaMC"
METRIC = "top_logit" if "70B" in BASE_MODEL_NAME else "entropy"

# Explicit path to directions file (bypasses auto-discovery if set)
# Set to None to use automatic pattern matching based on config below
DIRECTIONS_FILE = "Llama-3.1-8B-Instruct_TriviaMC_introspection_entropy_directions.npz"

# Optional filters for finding direction files (only used when DIRECTIONS_FILE is None)
META_TASK = None  
DIRECTION_TYPE = None 

# Quantization options for large models
LOAD_IN_4BIT = True if "70B" in BASE_MODEL_NAME else False
LOAD_IN_8BIT = False

# Optimization: Batch Size
# 8-16 is usually safe for 8B models on 24GB VRAM.
BATCH_SIZE = 8

# Confidence testing options
RUN_CONFIDENCE_TESTS = True  # Enable log probability tracking (minimal overhead)
RUN_BASELINE_COMPARISON = True  # Compare against random vectors (slower, ~3x per layer)
NUM_BASELINE_VECTORS = 3  # Number of random baselines if comparison enabled

# Semantic analysis options
RUN_SEMANTIC_ANALYSIS = True  # Enable semantic similarity analysis
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, fast

# Target concepts for semantic similarity analysis (3 core concepts)
TARGET_CONCEPTS = {
    "introspection": [
        "the model examining its own thoughts",
        "self-reflection about internal processes",
        "looking inward at own reasoning",
        "thinking about my own thoughts",
        "examining internal cognitive states",
        "metacognition",
        "self-awareness",
        "self-referential thinking",
        "introspection",
    ],
    "confidence": [
        "certainty about the answer",
        "conviction that response is correct",
        "high confidence in knowledge",
        "sure of being correct",
        "strong belief in the answer",
        "confident",
    ],
    "uncertainty": [
        "doubt",
        "unsure",
        "hesitiation",
        "uncertainty",
        "ambiguity",
    ],
}

# Output directory
OUTPUTS_DIR = Path("outputs")

# Activation Oracle adapter paths
AO_ADAPTERS = {
    "8b": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
    "70b": "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
}

# Model configurations
MODEL_CONFIGS = {
    "8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "ao_adapter": AO_ADAPTERS["8b"],
        "num_layers": 32,
        "d_model": 4096,
    },
    "70b": {
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "ao_adapter": AO_ADAPTERS["70b"],
        "num_layers": 80,
        "d_model": 8192,
    },
}

PLACEHOLDER_TOKEN = " ?"

# Questions to ask the activation oracle
INTERPRETATION_QUESTIONS = [
    "What concept does this represent?",
    "Is this related to confidence, uncertainty, introspection/self-reflection, or something else?",
    "What type of mental process or state does this encode?",
]

# Multiple-choice option mapping for Q2 (maps response keywords to target concepts)
MC_OPTION_MAPPING = {
    "confidence": "confidence",
    "certainty": "uncertainty",
    "uncertainty": "uncertainty",
    "introspection": "introspection",
    "self-reflection": "introspection",
    "something else": "other",
}

# Fuzzy keyword patterns for concept matching
# Refined keywords based on actual good AO responses
CONCEPT_KEYWORDS = {
    "introspection": [
        "introspect", "self-reflect", "self-aware", "self-referent",
        "own thought", "own reasoning", "own mental", "own thinking",
        "looking inward", "contemplat", "examine itself",
        "internal process", "inner state", "self-examin", "meta-cognit",
    ],
    "confidence": [
        "confident", "sure", "conviction",
        "knows the answer", "definite", "assured",
        "high confidence", "strong belief",
    ],
    "uncertainty": [
        "uncertain", "unsure", "doubt",
        "hesitat", "unclear", "not confident",
        "not sure", "ambigu",
    ],
}

# =============================================================================
# Semantic Analysis Infrastructure
# =============================================================================

# Lazy-loaded embedding model (only loaded when needed)
_embedding_model = None
_concept_embeddings = None
_sentence_transformers_available = None  # None = not checked yet


def _check_sentence_transformers():
    """Check if sentence-transformers is available, warn once if not."""
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
            print("WARNING: sentence-transformers not installed. Using fuzzy keyword matching instead.")
            print("         Install with: pip install sentence-transformers")
    return _sentence_transformers_available


def get_embedding_model():
    """Lazy-load the sentence embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_concept_embeddings() -> Dict[str, np.ndarray]:
    """Get pre-computed embeddings for target concepts (cached)."""
    global _concept_embeddings
    if _concept_embeddings is None:
        model = get_embedding_model()
        _concept_embeddings = {}
        for concept, exemplars in TARGET_CONCEPTS.items():
            # Embed all exemplars for this concept
            embs = model.encode(exemplars, normalize_embeddings=True)
            _concept_embeddings[concept] = embs
    return _concept_embeddings


def compute_fuzzy_concept_similarity(response: str) -> Dict[str, float]:
    """
    Fallback: compute concept similarity using fuzzy keyword matching.
    Returns dict mapping concept name to match score (0-1).
    """
    response_lower = response.lower()
    similarities = {}

    for concept, keywords in CONCEPT_KEYWORDS.items():
        # Count keyword matches (partial matching via 'in')
        matches = sum(1 for kw in keywords if kw.lower() in response_lower)
        # Normalize by number of keywords, cap at 1.0
        # Scale so that 2+ matches gives high similarity
        score = min(1.0, matches / 2.0) if matches > 0 else 0.0
        similarities[concept] = score

    return similarities


def get_keyword_matches(response: str) -> Dict[str, List[str]]:
    """
    Return the actual keywords that matched for each concept.
    Useful for understanding why a response was classified.
    """
    response_lower = response.lower()
    matches = {}

    for concept, keywords in CONCEPT_KEYWORDS.items():
        matched = [kw for kw in keywords if kw.lower() in response_lower]
        matches[concept] = matched

    return matches


def compute_semantic_similarity(response: str) -> Dict[str, float]:
    """
    Compute similarity of response to each target concept cluster.
    Uses sentence embeddings if available, falls back to fuzzy keyword matching.
    Returns dict mapping concept name to similarity score (0-1).
    """
    # Check if sentence-transformers is available
    if not _check_sentence_transformers():
        return compute_fuzzy_concept_similarity(response)

    model = get_embedding_model()
    concept_embs = get_concept_embeddings()

    # Embed the response
    response_emb = model.encode(response, normalize_embeddings=True)

    similarities = {}
    for concept, exemplar_embs in concept_embs.items():
        # Cosine similarity to each exemplar (embeddings are normalized)
        sims = exemplar_embs @ response_emb
        # Take max similarity to any exemplar in cluster
        ###similarities[concept] = float(np.max(sims))
        # Aggregate similarity over the closest few exemplars (less brittle than max)
        k = min(3, sims.shape[0])
        topk = np.partition(sims, -k)[-k:]
        similarities[concept] = float(np.mean(topk))

    return similarities


def extract_mc_choice(response: str) -> str:
    """
    Extract the selected option from a multiple-choice response (Q2).

    Looks for option keywords in the response and maps to target concepts.
    Returns the mapped concept or "other" if no clear match.
    """
    response_lower = response.lower()

    # Check for exact option phrases first (more specific)
    if "something else" in response_lower:
        return "other"

    # Check for key option words
    for keyword, concept in MC_OPTION_MAPPING.items():
        if keyword in response_lower:
            return concept

    return "other"


def compute_semantic_distinctiveness(
    probe_response: str,
    baseline_responses: List[str],
) -> Dict[str, Any]:
    """
    Compare probe response semantically to baseline responses.
    Returns distinctiveness score and per-concept comparison.
    Works with both sentence embeddings and fuzzy keyword fallback.
    """
    if not baseline_responses:
        return {"distinctiveness": None, "concept_delta": {}}

    # Per-concept: probe similarity vs baseline similarities
    # This works with both embedding-based and keyword-based similarity
    probe_concepts = compute_semantic_similarity(probe_response)
    baseline_concepts = [compute_semantic_similarity(b) for b in baseline_responses]

    concept_delta = {}
    for concept in TARGET_CONCEPTS:
        probe_sim = probe_concepts[concept]
        baseline_sims_concept = [b[concept] for b in baseline_concepts]
        mean_baseline = np.mean(baseline_sims_concept)
        concept_delta[concept] = float(probe_sim - mean_baseline)

    # Overall distinctiveness: use embeddings if available, otherwise use concept divergence
    if _check_sentence_transformers():
        model = get_embedding_model()
        probe_emb = model.encode(probe_response, normalize_embeddings=True)
        baseline_embs = model.encode(baseline_responses, normalize_embeddings=True)
        baseline_sims = baseline_embs @ probe_emb
        distinctiveness = 1.0 - float(np.mean(baseline_sims))
    else:
        # Fallback: use mean absolute concept delta as distinctiveness proxy
        distinctiveness = float(np.mean([abs(d) for d in concept_delta.values()]))

    return {
        "distinctiveness": distinctiveness,
        "probe_concepts": probe_concepts,
        "concept_delta": concept_delta,
    }


def compute_entropy_metrics(scores: List) -> Dict[str, float]:
    """
    Compute entropy-based confidence metrics from generation scores.

    Args:
        scores: List of score tensors from model.generate(output_scores=True)
                Each tensor has shape (batch_size, vocab_size)

    Returns:
        Dict with mean_entropy, max_entropy, entropy_trend, mean_top_k_concentration
    """
    import torch

    if not scores:
        return {
            "mean_entropy": None,
            "max_entropy": None,
            "entropy_trend": None,
            "mean_top_k_concentration": None,
        }

    entropies = []
    top_k_concentrations = []

    for score in scores:
        # Handle batched scores - take first item if batched
        if score.dim() > 1:
            score = score[0]

        # Compute probabilities
        probs = torch.softmax(score.float(), dim=-1)

        # Entropy: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs).item()
        entropies.append(entropy)

        # Top-k concentration (k=10)
        top_k_mass = torch.topk(probs, k=10).values.sum().item()
        top_k_concentrations.append(top_k_mass)

    return {
        "mean_entropy": float(np.mean(entropies)),
        "max_entropy": float(np.max(entropies)),
        "entropy_trend": float(entropies[-1] - entropies[0]) if len(entropies) > 1 else 0.0,
        "mean_top_k_concentration": float(np.mean(top_k_concentrations)),
    }


def compute_entropy_quality(entropy_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Compute entropy-based quality indicators.

    Key insight: The ratio of max_entropy / mean_entropy discriminates good from garbage responses.
    - Garbage (repetitive text like "1.0 0.0 0.0..."): ratio > 20 (one entropy spike, then very low)
    - Good responses: ratio < 5 (consistent moderate entropy throughout)

    Args:
        entropy_data: Dict with mean_entropy, max_entropy from generation

    Returns:
        Dict with entropy_consistency_ratio, is_likely_garbage, is_consistent
    """
    mean_ent = entropy_data.get("mean_entropy")
    max_ent = entropy_data.get("max_entropy")

    if mean_ent is None or max_ent is None:
        return {
            "entropy_consistency_ratio": None,
            "is_likely_garbage": None,
            "is_consistent": None,
        }

    if mean_ent < 0.01:
        ratio = float('inf')
    else:
        ratio = max_ent / mean_ent

    return {
        "entropy_consistency_ratio": ratio,
        "is_likely_garbage": ratio > 20,  # High ratio = garbage
        "is_consistent": ratio < 5,  # Low ratio = thoughtful
    }


def entropy_ratio_to_quality(ratio: float) -> float:
    """
    Convert entropy consistency ratio to a 0-1 quality score.

    Low ratio (consistent generation) → high quality
    High ratio (one spike then repetitive) → low quality

    Calibrated so:
    - ratio < 5 → quality > 0.7 (high quality)
    - ratio 5-20 → quality 0.4-0.7 (ambiguous)
    - ratio > 20 → quality < 0.4 (garbage)
    """
    if ratio is None or ratio == float('inf'):
        return 0.0
    if ratio <= 0:
        return 0.0
    # Using log1p for smooth decay: 1 / (1 + ln(1 + ratio/5))
    return 1.0 / (1.0 + np.log1p(ratio / 5.0))


# =============================================================================
# AO Output Analysis Functions
# =============================================================================

def detect_repetition(text: str, ngram_size: int = 4) -> float:
    """
    Detect repetitive/looping text by measuring n-gram repetition.
    Returns 0-1 score (higher = more repetitive).
    """
    if len(text) < ngram_size * 2:
        return 0.0

    words = text.lower().split()
    if len(words) < ngram_size * 2:
        return 0.0

    # Create n-grams
    ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]

    if len(ngrams) == 0:
        return 0.0

    # Count occurrences
    ngram_counts = Counter(ngrams)

    # Calculate repetition: ratio of repeated ngrams to total
    repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
    total = len(ngrams)

    return min(1.0, repeated / total) if total > 0 else 0.0


def compute_coherence_metrics(response: str) -> Dict[str, float]:
    """
    Compute metrics indicating response quality.
    """
    # Basic counts
    length = len(response)
    words = response.split()
    word_count = len(words)

    # Sentence count (rough approximation)
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Repetition detection
    repetition_score = detect_repetition(response)

    # On-topic score: how many concept categories are represented
    concept_sims = compute_fuzzy_concept_similarity(response)
    concepts_present = sum(1 for score in concept_sims.values() if score > 0)
    on_topic_score = concepts_present / len(CONCEPT_KEYWORDS)

    # Average sentence length
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    return {
        "length": length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "repetition_score": repetition_score,
        "on_topic_score": on_topic_score,
        "avg_sentence_length": avg_sentence_length,
    }


def compute_quality_score(metrics: Dict[str, float]) -> float:
    """
    Compute overall quality score from coherence metrics.
    Higher score = better quality response.
    """
    score = 1.0

    # Length penalty: too short is bad
    if metrics["word_count"] < 20:
        score *= 0.5
    elif metrics["word_count"] < 50:
        score *= 0.8

    # Length penalty: too long may indicate rambling
    if metrics["word_count"] > 500:
        score *= 0.7

    # Repetition penalty
    score *= (1.0 - metrics["repetition_score"])

    # On-topic bonus
    score *= (0.5 + 0.5 * metrics["on_topic_score"])

    # Sentence structure: very short or very long sentences are bad
    if metrics["avg_sentence_length"] < 5:
        score *= 0.7
    elif metrics["avg_sentence_length"] > 50:
        score *= 0.7

    return score


def analyze_ao_outputs(
    results: Dict[int, Dict[str, str]],
    baseline_results: Dict[int, Dict] = None,
) -> Dict:
    """
    Analyze all AO outputs using entropy quality gating and keyword matching.

    Key improvement: Uses entropy consistency ratio (max_entropy / mean_entropy) to
    identify garbage responses before semantic analysis. High ratio (>20) indicates
    repetitive garbage; low ratio (<5) indicates consistent, thoughtful responses.
    """
    per_layer = {}
    layer_quality = {}
    layer_confidence = {}
    layer_entropy = {}
    layer_semantic = {}
    question_quality = {q: [] for q in INTERPRETATION_QUESTIONS}
    question_confidence = {q: [] for q in INTERPRETATION_QUESTIONS}

    # Track quality counts across all responses
    quality_counts = {"high_quality": 0, "garbage": 0, "ambiguous": 0, "total": 0}

    # Track concept signals from high-quality responses only
    concept_signals = {c: [] for c in TARGET_CONCEPTS}  # List of (layer, question, match_count, snippet, keywords)

    for layer_idx, layer_results in results.items():
        layer_analysis = {}
        layer_scores = []
        layer_logprobs = []
        layer_entropies = []
        layer_concept_sims = {c: [] for c in TARGET_CONCEPTS}

        for question, result in layer_results.items():
            # Handle both confidence dict and plain string responses
            if isinstance(result, dict):
                response = result["response"]
                mean_logprob = result.get("mean_logprob")
                min_logprob = result.get("min_logprob")
                mean_entropy = result.get("mean_entropy")
                max_entropy = result.get("max_entropy")
                entropy_trend = result.get("entropy_trend")
                mean_top_k = result.get("mean_top_k_concentration")
            else:
                response = result
                mean_logprob = None
                min_logprob = None
                mean_entropy = None
                max_entropy = None
                entropy_trend = None
                mean_top_k = None

            metrics = compute_coherence_metrics(response)
            quality = compute_quality_score(metrics)

            # Compute entropy quality indicators
            entropy_data = {"mean_entropy": mean_entropy, "max_entropy": max_entropy}
            entropy_quality = compute_entropy_quality(entropy_data)

            layer_analysis[question] = {
                "metrics": metrics,
                "quality": quality,
                "entropy_quality": entropy_quality,
            }

            # Track quality counts
            quality_counts["total"] += 1
            if entropy_quality["is_consistent"]:
                quality_counts["high_quality"] += 1
            elif entropy_quality["is_likely_garbage"]:
                quality_counts["garbage"] += 1
            else:
                quality_counts["ambiguous"] += 1

            # Add confidence if available
            if mean_logprob is not None:
                layer_analysis[question]["confidence"] = {
                    "mean_logprob": mean_logprob,
                    "min_logprob": min_logprob,
                }
                layer_logprobs.append(mean_logprob)
                if question in question_confidence:
                    question_confidence[question].append(mean_logprob)

            # Add entropy metrics if available
            if mean_entropy is not None:
                layer_analysis[question]["entropy"] = {
                    "mean_entropy": mean_entropy,
                    "max_entropy": max_entropy,
                    "entropy_trend": entropy_trend,
                    "mean_top_k_concentration": mean_top_k,
                }
                layer_entropies.append(mean_entropy)

            # Get keyword matches for concept detection
            keyword_matches = get_keyword_matches(response)
            layer_analysis[question]["keyword_matches"] = keyword_matches

            # For Q2 (multiple choice), extract the selected option
            if question == INTERPRETATION_QUESTIONS[1]:
                mc_choice = extract_mc_choice(response)
                layer_analysis[question]["mc_choice"] = mc_choice

            # Only analyze semantic similarity for high-quality responses
            is_high_quality = entropy_quality.get("is_consistent", False) or (
                entropy_quality.get("entropy_consistency_ratio") is not None and
                entropy_quality["entropy_consistency_ratio"] < 10  # slightly relaxed threshold
            )

            if RUN_SEMANTIC_ANALYSIS:
                concept_sims = compute_semantic_similarity(response)
                layer_analysis[question]["semantic"] = concept_sims

                # Only include in layer average if high quality
                if is_high_quality:
                    for concept, sim in concept_sims.items():
                        layer_concept_sims[concept].append(sim)

            # Track concept signals from high-quality responses
            if is_high_quality:
                for concept, matched_kws in keyword_matches.items():
                    if matched_kws:  # Has keyword matches
                        # Get first 100 chars as snippet
                        snippet = response[:100].replace("\n", " ")
                        if len(response) > 100:
                            snippet += "..."
                        concept_signals[concept].append({
                            "layer": layer_idx,
                            "question": question,
                            "match_count": len(matched_kws),
                            "keywords": matched_kws,
                            "snippet": snippet,
                            "entropy_ratio": entropy_quality.get("entropy_consistency_ratio"),
                        })

            layer_scores.append(quality)

            # Track per-question quality
            if question in question_quality:
                question_quality[question].append(quality)

        per_layer[layer_idx] = layer_analysis
        layer_quality[layer_idx] = np.mean(layer_scores) if layer_scores else 0.0
        if layer_logprobs:
            layer_confidence[layer_idx] = np.mean(layer_logprobs)
        if layer_entropies:
            layer_entropy[layer_idx] = np.mean(layer_entropies)

        # Average semantic similarity per concept for this layer (high-quality only)
        if any(layer_concept_sims[c] for c in TARGET_CONCEPTS):
            layer_semantic[layer_idx] = {
                c: np.mean(sims) if sims else 0.0
                for c, sims in layer_concept_sims.items()
            }

    # Sort layers by quality
    best_layers = sorted(layer_quality.items(), key=lambda x: x[1], reverse=True)

    # Average quality per question
    question_avg_quality = {
        q: np.mean(scores) if scores else 0.0
        for q, scores in question_quality.items()
    }

    # Average confidence per question
    question_avg_confidence = {
        q: np.mean(scores) if scores else None
        for q, scores in question_confidence.items()
    }

    result = {
        "per_layer": per_layer,
        "layer_quality": layer_quality,
        "best_layers": best_layers,
        "question_quality": question_avg_quality,
        "quality_counts": quality_counts,
        "concept_signals": concept_signals,
    }

    # Add confidence data if available
    if layer_confidence:
        result["layer_confidence"] = layer_confidence
        result["question_confidence"] = question_avg_confidence

    # Add entropy data if available
    if layer_entropy:
        result["layer_entropy"] = layer_entropy

    # Add semantic analysis if available (filtered by quality)
    if layer_semantic:
        result["layer_semantic"] = layer_semantic

    # Add baseline summary if available
    if baseline_results:
        baseline_summary = {}
        for layer_idx, baseline in baseline_results.items():
            baseline_summary[layer_idx] = {
                "probe_logprob": baseline["probe_logprob"],
                "baseline_mean_logprob": baseline["baseline_mean_logprob"],
                "logprob_delta": baseline["logprob_delta"],
            }
            # Add semantic distinctiveness if we have baseline responses
            if RUN_SEMANTIC_ANALYSIS and "baseline_results" in baseline:
                probe_response = results[layer_idx][INTERPRETATION_QUESTIONS[0]]
                if isinstance(probe_response, dict):
                    probe_response = probe_response["response"]
                baseline_responses = baseline["baseline_results"]
                if baseline_responses:
                    semantic_dist = compute_semantic_distinctiveness(probe_response, baseline_responses)
                    baseline_summary[layer_idx]["semantic_distinctiveness"] = semantic_dist["distinctiveness"]
                    baseline_summary[layer_idx]["concept_delta"] = semantic_dist["concept_delta"]
        result["baseline_summary"] = baseline_summary

    # ==========================================================================
    # Generate unified layer summary
    # ==========================================================================

    # Build per-layer summary with quality, distinctiveness, theme, keywords
    layer_summary = []
    baseline_summary_data = result.get("baseline_summary", {})

    for layer_idx in sorted(results.keys()):
        # Compute average quality score from entropy ratios across questions
        layer_data = per_layer.get(layer_idx, {})
        ratios = []
        for q_data in layer_data.values():
            eq = q_data.get("entropy_quality", {})
            ratio = eq.get("entropy_consistency_ratio")
            if ratio is not None and ratio != float('inf'):
                ratios.append(ratio)

        if ratios:
            avg_ratio = np.mean(ratios)
            quality_score = entropy_ratio_to_quality(avg_ratio)
        else:
            avg_ratio = None
            quality_score = 0.0

        # Quality indicator
        if quality_score >= 0.7:
            quality_indicator = "✓"
        elif quality_score >= 0.4:
            quality_indicator = "~"
        else:
            quality_indicator = "✗"

        # Distinctiveness from baseline comparison
        # Use semantic_distinctiveness (float), not concept_delta (dict)
        distinctiveness = baseline_summary_data.get(layer_idx, {}).get("semantic_distinctiveness", 0.0)
        if distinctiveness is None or not isinstance(distinctiveness, (int, float)):
            distinctiveness = 0.0

        # Theme: concept with highest embedding similarity
        layer_sem = layer_semantic.get(layer_idx, {})
        if layer_sem:
            best_concept = max(layer_sem.keys(), key=lambda c: layer_sem[c])
            best_sim = layer_sem[best_concept]
        else:
            best_concept = "none"
            best_sim = 0.0

        # Keywords matched for the best concept
        matched_keywords = []
        for signal in concept_signals.get(best_concept, []):
            if signal["layer"] == layer_idx:
                matched_keywords.extend(signal["keywords"])
        matched_keywords = list(set(matched_keywords))[:4]  # Dedupe and limit

        # Get MC choice from Q2 if available
        mc_choice = None
        q2 = INTERPRETATION_QUESTIONS[1]
        if q2 in layer_data:
            mc_choice = layer_data[q2].get("mc_choice")

        layer_summary.append({
            "layer": layer_idx,
            "quality": round(quality_score, 2),
            "quality_indicator": quality_indicator,
            "distinctiveness": round(distinctiveness, 2) if distinctiveness else 0.0,
            "theme": best_concept,
            "theme_score": round(best_sim, 3),
            "keywords": matched_keywords,
            "mc_choice": mc_choice,  # Direct concept selection from Q2
            "avg_entropy_ratio": round(avg_ratio, 1) if avg_ratio else None,
        })

    # Sort by quality for the summary (keep layer_summary_by_quality separate)
    layer_summary_by_quality = sorted(layer_summary, key=lambda x: -x["quality"])

    # Top layers per concept
    top_layers_by_concept = {}
    for concept in TARGET_CONCEPTS:
        # Get layers sorted by embedding score for this concept
        concept_layers = []
        for ls in layer_summary:
            layer_idx = ls["layer"]
            emb_score = layer_semantic.get(layer_idx, {}).get(concept, 0)
            concept_layers.append((layer_idx, emb_score))
        concept_layers.sort(key=lambda x: -x[1])
        top_layers_by_concept[concept] = [l[0] for l in concept_layers[:5]]

    summary = {
        "quality_counts": {
            "total_responses": quality_counts["total"],
            "high_quality": quality_counts["high_quality"],
            "garbage": quality_counts["garbage"],
            "ambiguous": quality_counts["ambiguous"],
        },
        "layer_summary": layer_summary_by_quality,  # Sorted by quality
        "top_layers_by_concept": top_layers_by_concept,
    }

    result["summary"] = summary

    return result


def generate_text_summary(analysis: Dict, output_path: str) -> None:
    """
    Write human-readable summary to text file.

    Creates a formatted text file with:
    - Response quality counts
    - Layer summary table (sorted by quality)
    - Top layers per concept
    """
    summary = analysis.get("summary", {})
    qc = summary.get("quality_counts", {})
    layer_summary = summary.get("layer_summary", [])
    top_layers = summary.get("top_layers_by_concept", {})

    lines = []
    lines.append("AO INTERPRETATION SUMMARY")
    lines.append("=" * 77)
    lines.append("")

    # Quality counts
    total = qc.get("total_responses", 0)
    high_q = qc.get("high_quality", 0)
    garbage = qc.get("garbage", 0)
    ambig = qc.get("ambiguous", 0)
    lines.append(f"Response Quality: {high_q}/{total} high-quality, {garbage} garbage, {ambig} ambiguous")
    lines.append("")

    # Layer summary table
    lines.append("LAYER SUMMARY (sorted by quality)")
    lines.append("=" * 95)
    lines.append(f"{'Layer':<6} {'Quality':<9} {'Distinct.':<10} {'MC Choice':<14} {'Theme':<14} {'Score':<7} {'Keywords'}")
    lines.append("-" * 95)

    for ls in layer_summary:
        layer = ls["layer"]
        quality = ls["quality"]
        indicator = ls["quality_indicator"]
        distinct = ls["distinctiveness"]
        theme = ls["theme"]
        theme_score = ls["theme_score"]
        mc_choice = ls.get("mc_choice", "-") or "-"
        keywords = ", ".join(ls["keywords"][:3]) if ls["keywords"] else ""

        lines.append(
            f"{layer:>4}   {quality:.2f} {indicator:<2}  {distinct:>6.2f}     "
            f"{mc_choice:<14} {theme:<14} {theme_score:<7.3f} {keywords}"
        )

    lines.append("")
    lines.append("")

    # Top layers by concept
    lines.append("TOP LAYERS BY CONCEPT")
    lines.append("=" * 77)
    for concept, layers in top_layers.items():
        layer_str = ", ".join(str(l) for l in layers)
        lines.append(f"{concept.upper()}: {layer_str}")

    lines.append("")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# =============================================================================
# Path utilities
# =============================================================================

def get_model_prefix() -> str:
    """Get model prefix for filenames."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return f"{model_short}_adapter-{adapter_short}"
    return model_short


def find_directions_file() -> Path:
    """
    Find available directions file.
    """
    if DIRECTIONS_FILE is not None:
        path = Path(DIRECTIONS_FILE)
        if not path.is_absolute():
            path = OUTPUTS_DIR / path
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified directions file not found: {path}")

    model_prefix = get_model_prefix()
    task_suffix = f"_{META_TASK}" if META_TASK else ""

    patterns = []
    if DIRECTION_TYPE:
        patterns.append(f"{model_prefix}_{DATASET_NAME}_{METRIC}{task_suffix}_{DIRECTION_TYPE}_directions.npz")
    else:
        for dt in ["calibration", "contrastive"]:
            patterns.append(f"{model_prefix}_{DATASET_NAME}_{METRIC}{task_suffix}_{dt}_directions.npz")

    patterns.append(f"{model_prefix}_{DATASET_NAME}_introspection{task_suffix}_{METRIC}_directions.npz")
    patterns.append(f"{model_prefix}_{DATASET_NAME}_introspection{task_suffix}_{METRIC}_probe_directions.npz")
    patterns.append(f"{model_prefix}_{DATASET_NAME}_mc_{METRIC}_directions.npz")
    patterns.append(f"{model_prefix}_{DATASET_NAME}_{METRIC}_directions.npz")

    for pattern in patterns:
        path = OUTPUTS_DIR / pattern
        if path.exists():
            return path

    glob_patterns = [
        f"{model_prefix}*{METRIC}*directions.npz",
        f"{model_prefix}*directions.npz",
    ]

    for pattern in glob_patterns:
        matches = list(OUTPUTS_DIR.glob(pattern))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    raise FileNotFoundError(
        f"No directions file found in {OUTPUTS_DIR}\n"
        f"Model: {model_prefix}, Dataset: {DATASET_NAME}, Metric: {METRIC}"
    )


def load_directions(path: Path) -> dict:
    """Load direction vectors from npz file."""
    if not path.exists():
        raise FileNotFoundError(f"Directions file not found: {path}")

    data = np.load(path)
    directions = {}
    for key in data.files:
        if key.startswith("_"):
            continue
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "layer":
            layer_idx = int(parts[1])
            directions[layer_idx] = data[key]
    return directions


# =============================================================================
# Activation Oracle Interpreter (OPTIMIZED)
# =============================================================================

class ProbeInterpreter:
    def __init__(
        self,
        model_size: str = "70b",
        load_in_4bit: bool = LOAD_IN_4BIT,
        load_in_8bit: bool = LOAD_IN_8BIT,
        device: str = "auto",
    ):
        """
        Initialize the Activation Oracle for interpreting probe directions.
        """
        import torch
        from peft import PeftModel
        from core import load_model_and_tokenizer

        self.torch = torch

        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")

        self.config = MODEL_CONFIGS[model_size]
        self.device = device

        # Load base model with quantization
        print(f"Loading base model: {self.config['base_model']}")
        self.model, self.tokenizer, _ = load_model_and_tokenizer(
            self.config["base_model"],
            adapter_path=None, 
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        # Load AO adapter on top
        print(f"Loading AO adapter: {self.config['ao_adapter']}")
        self.model = PeftModel.from_pretrained(self.model, self.config["ao_adapter"])
        self.model.eval()

        # IMPORTANT: Set padding side to left for batched generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find placeholder token id
        #self.placeholder_id = self.tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)[0]
        ids = self.tokenizer.encode(PLACEHOLDER_TOKEN, add_special_tokens=False)
        assert len(ids) == 1, f"PLACEHOLDER_TOKEN tokenizes to {ids}"
        self.placeholder_id = ids[0]

        print("Ready!")

    def _make_batch_injection_hook(self, vectors, placeholder_indices):
        """
        Inject vectors into specific batch indices at specific sequence positions.
        vectors: Tensor (batch, hidden_dim)
        placeholder_indices: List[int] of positions for each batch item
        """
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # hidden_states: (batch, seq_len, hidden_dim)
            batch_size = hidden_states.shape[0]

            for b in range(batch_size):
                if b >= len(placeholder_indices) or b >= vectors.shape[0]:
                    continue
                
                pos = int(placeholder_indices[b])
                if pos >= hidden_states.shape[1]:
                    continue

                v = vectors[b].to(hidden_states.device, hidden_states.dtype)
                h = hidden_states[b, pos].clone()

                # Norm-matched addition
                v_normalized = v / (v.norm() + 1e-8)
                h_norm = h.norm()
                new_h = h + h_norm * v_normalized
                
                hidden_states[b, pos, :] = new_h

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook

    def _get_layer_1_module(self):
        """Get the module after which to inject (layer 1)."""
        return self.model.base_model.model.model.layers[0]

    def interpret_batch(
        self,
        tasks: List[Dict],
        max_new_tokens: int = 256,
        with_confidence: bool = False
    ) -> List[Dict]:
        """
        Interpret a batch of directions.
        tasks: List of dicts, each containing:
               {'vector': Tensor, 'prompt': str, 'layer': int, 'metadata': Any}
        """
        if not tasks:
            return []

        # Prepare batch inputs
        prompts = [t['prompt'] for t in tasks]
        vectors = self.torch.stack([t['vector'] for t in tasks])

        # Tokenize (Left Padding)
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Find placeholder positions for each batch item
        placeholder_indices = []
        for i in range(input_ids.shape[0]):
            matches = (input_ids[i] == self.placeholder_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                # Use the last occurrence if multiple (standard behavior)
                placeholder_indices.append(matches[-1].item())
            else:
                placeholder_indices.append(0) # Should not happen if prompts are correct

        # Register Hook
        layer_1 = self._get_layer_1_module()
        hook = self._make_batch_injection_hook(vectors, placeholder_indices)
        handle = layer_1.register_forward_hook(hook)

        results = []
        try:
            with self.torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=with_confidence,
                )

            # Process outputs
            gen_ids = outputs.sequences[:, input_ids.shape[1]:]
            
            for i, task in enumerate(tasks):
                response = self.tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
                
                result_entry = {
                    "response": response,
                    "layer": task['layer'],
                    "metadata": task.get('metadata')
                }

                if with_confidence and outputs.scores:
                    # Calculate logprobs for this sequence
                    logprobs = []
                    tokens = []
                    entropies = []
                    top_k_concentrations = []

                    # Iterate through steps (seq_len)
                    for step_idx, step_scores in enumerate(outputs.scores):
                        if step_idx >= len(gen_ids[i]):
                            break

                        token_id = gen_ids[i][step_idx].item()
                        if token_id == self.tokenizer.pad_token_id:
                            continue

                        # Step scores is (batch, vocab)
                        probs = self.torch.softmax(step_scores[i].float(), dim=-1)
                        lp = self.torch.log(probs[token_id] + 1e-10).item()
                        logprobs.append(lp)
                        tokens.append(self.tokenizer.decode([token_id]))

                        # Compute entropy: -sum(p * log(p))
                        log_probs_all = self.torch.log(probs + 1e-10)
                        entropy = -self.torch.sum(probs * log_probs_all).item()
                        entropies.append(entropy)

                        # Top-k concentration
                        top_k_mass = self.torch.topk(probs, k=10).values.sum().item()
                        top_k_concentrations.append(top_k_mass)

                    if logprobs:
                        result_entry["mean_logprob"] = sum(logprobs) / len(logprobs)
                        result_entry["min_logprob"] = min(logprobs)
                    else:
                        result_entry["mean_logprob"] = 0.0
                        result_entry["min_logprob"] = 0.0

                    # Add entropy metrics
                    if entropies:
                        result_entry["mean_entropy"] = sum(entropies) / len(entropies)
                        result_entry["max_entropy"] = max(entropies)
                        result_entry["entropy_trend"] = entropies[-1] - entropies[0] if len(entropies) > 1 else 0.0
                        result_entry["mean_top_k_concentration"] = sum(top_k_concentrations) / len(top_k_concentrations)
                    else:
                        result_entry["mean_entropy"] = None
                        result_entry["max_entropy"] = None
                        result_entry["entropy_trend"] = None
                        result_entry["mean_top_k_concentration"] = None

                    result_entry["tokens"] = tokens
                    
                results.append(result_entry)

        finally:
            handle.remove()

        return results


# =============================================================================
# Visualization
# =============================================================================

def visualize_ao_results(
    results: Dict[int, Dict[str, str]],
    analysis: Dict,
    output_path: str,
    title_suffix: str = "",
):
    """
    Create a simplified 2-panel visualization:
    - Panel 1: Concept heatmap with quality annotations (layers in order)
    - Panel 2: Quality & distinctiveness line chart (layers in order)
    """
    layers = sorted(results.keys())
    n_layers = len(layers)

    if n_layers == 0:
        print("No results to plot")
        return

    # Get layer summary for quality/distinctiveness data
    layer_summary_data = analysis.get("summary", {}).get("layer_summary", [])
    # Create lookup by layer index
    summary_by_layer = {ls["layer"]: ls for ls in layer_summary_data}

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # =========================================================================
    # Panel 1: Concept Heatmap by Layer (with quality annotations)
    # =========================================================================
    ax1 = axes[0]

    concepts = list(TARGET_CONCEPTS.keys())
    semantic_matrix = np.zeros((n_layers, len(concepts)))

    for i, layer_idx in enumerate(layers):
        layer_sem = analysis.get("layer_semantic", {}).get(layer_idx, {})
        for j, concept in enumerate(concepts):
            semantic_matrix[i, j] = layer_sem.get(concept, 0.0)

    # Create heatmap
    sns.heatmap(
        semantic_matrix,
        ax=ax1,
        cmap="YlOrRd",
        vmin=0, vmax=0.6,  # Typical range for embedding similarity
        xticklabels=[c.title() for c in concepts],
        yticklabels=[f"L{l}" for l in layers],
        cbar_kws={"label": "Embedding Similarity"},
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
    )

    # Add quality indicators as row labels on the right
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(ax1.get_ylim())
    ax1_twin.set_yticks(np.arange(n_layers) + 0.5)

    quality_labels = []
    for layer_idx in layers:
        ls = summary_by_layer.get(layer_idx, {})
        indicator = ls.get("quality_indicator", "?")
        quality_labels.append(indicator)

    ax1_twin.set_yticklabels(quality_labels, fontsize=10)
    ax1_twin.tick_params(axis='y', length=0)

    ax1.set_title("Concept Similarity by Layer\n(✓=high quality, ~=ambiguous, ✗=garbage)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Concept")
    ax1.set_ylabel("Layer")

    # =========================================================================
    # Panel 2: Quality & Distinctiveness by Layer
    # =========================================================================
    ax2 = axes[1]

    # Extract quality and distinctiveness per layer (in layer order)
    layer_qualities = []
    layer_distincts = []

    for layer_idx in layers:
        ls = summary_by_layer.get(layer_idx, {})
        layer_qualities.append(ls.get("quality", 0.0))
        layer_distincts.append(ls.get("distinctiveness", 0.0))

    # Plot both metrics
    ax2.plot(layers, layer_qualities, 'o-', label='Quality (entropy consistency)', color='green', linewidth=2, markersize=6)
    ax2.plot(layers, layer_distincts, 's--', label='Distinctiveness (vs baseline)', color='purple', linewidth=2, markersize=5, alpha=0.8)

    # Add threshold lines
    ax2.axhline(0.7, color='green', linestyle=':', alpha=0.5, label='Quality threshold (✓)')
    ax2.axhline(0.4, color='orange', linestyle=':', alpha=0.5, label='Quality threshold (~)')

    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Score (0-1)")
    ax2.set_title("Layer Quality & Distinctiveness", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(min(layers) - 0.5, max(layers) + 0.5)

    # =========================================================================
    # Title and save
    # =========================================================================
    model_short = get_model_short_name(BASE_MODEL_NAME)
    suptitle = f"AO Interpretation: {model_short} - {DATASET_NAME}"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def plot_interpretations(
    results: dict,
    output_path: str,
    title_suffix: str = "",
):
    """
    DEPRECATED: Use visualize_ao_results() instead.

    Create a simple text dump of layer-by-layer interpretations.
    Kept for backward compatibility.
    """
    # Extract the primary question's responses
    primary_question = INTERPRETATION_QUESTIONS[0]

    layers = sorted(results.keys())
    n_layers = len(layers)

    if n_layers == 0:
        print("No results to plot")
        return

    # Create figure
    fig_height = max(8, n_layers * 0.4)
    _, ax = plt.subplots(figsize=(14, fig_height))

    # Plot each layer's interpretation
    for i, layer_idx in enumerate(layers):
        layer_results = results[layer_idx]
        if isinstance(layer_results, str):
            # Single question format
            interpretation = layer_results
        else:
            # Multi-question format
            interpretation = layer_results.get(primary_question, str(layer_results))

        # Truncate long interpretations
        max_chars = 120
        if len(interpretation) > max_chars:
            interpretation = interpretation[:max_chars] + "..."

        # Clean up whitespace
        interpretation = " ".join(interpretation.split())

        ax.text(
            0.02, i,
            f"Layer {layer_idx:2d}: {interpretation}",
            fontsize=8,
            va='center',
            fontfamily='monospace',
            wrap=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()  # Put layer 0 at top

    model_short = get_model_short_name(BASE_MODEL_NAME)
    title = f"Activation Oracle Interpretations\n{model_short} - {DATASET_NAME} - {DIRECTION_TYPE}"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def print_interpretations(results: dict, baseline_results: dict = None):
    """Print interpretations to console in a readable format."""
    print("\n" + "=" * 80)
    print("ACTIVATION ORACLE INTERPRETATIONS")
    print("=" * 80)

    for layer_idx in sorted(results.keys()):
        layer_results = results[layer_idx]
        print(f"\n{'─' * 80}")
        print(f"LAYER {layer_idx}")
        print("─" * 80)

        if isinstance(layer_results, str):
            print(f"  {layer_results}")
        else:
            for question, result in layer_results.items():
                print(f"\n  Q: {question}")
                # Handle both confidence dict and plain string responses
                if isinstance(result, dict):
                    print(f"  A: {result['response']}")
                    print(f"     (logprob: mean={result['mean_logprob']:.3f}, min={result['min_logprob']:.3f})")
                else:
                    print(f"  A: {result}")

        # Print baseline comparison if available
        if baseline_results and layer_idx in baseline_results:
            baseline = baseline_results[layer_idx]
            print(f"\n  BASELINE COMPARISON:")
            print(f"    Probe logprob: {baseline['probe_logprob']:.3f}")
            print(f"    Baseline mean: {baseline['baseline_mean_logprob']:.3f}")
            print(f"    Delta: {baseline['logprob_delta']:+.3f}")


# =============================================================================
# Main
# =============================================================================

def get_model_short_name(model_name: str) -> str:
    """Extract short name from model path (lazy import alternative)."""
    if "/" in model_name:
        return model_name.split("/")[-1]
    return model_name


def main():
    # Import heavy dependencies only when running full interpretation
    import torch
    from tqdm import tqdm

    print(f"\n{'=' * 70}")
    print("ACTIVATION ORACLE PROBE INTERPRETATION (ACCELERATED)")
    print("=" * 70)
    print(f"Model: {get_model_short_name(BASE_MODEL_NAME)}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Task: {META_TASK}")
    print(f"Direction type: {DIRECTION_TYPE}")
    print(f"Confidence testing: {RUN_CONFIDENCE_TESTS}")
    print(f"Baseline comparison: {RUN_BASELINE_COMPARISON} (n={NUM_BASELINE_VECTORS})")
    print(f"Semantic analysis: {RUN_SEMANTIC_ANALYSIS}")
    print(f"Batch Size: {BATCH_SIZE}")

    # Check for sentence-transformers if semantic analysis is enabled
    if RUN_SEMANTIC_ANALYSIS:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Embedding model: {EMBEDDING_MODEL}")
        except ImportError:
            print("  WARNING: sentence-transformers not installed. Semantic analysis will be skipped.")
            print("  Install with: pip install sentence-transformers")

    # Load directions
    try:
        directions_path = find_directions_file()
        print(f"\nLoading directions from: {directions_path}")
        directions = load_directions(directions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run run_contrastive_direction.py first to generate directions.")
        return

    print(f"Loaded directions for {len(directions)} layers: {sorted(directions.keys())}")

    # Determine model size from config
    if "70B" in BASE_MODEL_NAME or "70b" in BASE_MODEL_NAME:
        model_size = "70b"
    else:
        model_size = "8b"

    # Initialize interpreter
    print(f"\nInitializing interpreter (model_size={model_size})...")
    interpreter = ProbeInterpreter(
        model_size=model_size,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    layers_to_interpret = sorted(directions.keys())
    print(f"\nInterpreting {len(layers_to_interpret)} layers (Batched Execution)...")

    # =========================================================================
    # BATCH PREPARATION
    # =========================================================================
    
    tasks = []
    
    # 1. Main tasks (All layers x All questions)
    for layer_idx in layers_to_interpret:
        vector = torch.from_numpy(directions[layer_idx]).float()
        
        for question in INTERPRETATION_QUESTIONS:
            prompt = f"Layer {layer_idx}:{PLACEHOLDER_TOKEN} {question}"
            
            tasks.append({
                "type": "main",
                "layer": layer_idx,
                "vector": vector,
                "prompt": prompt,
                "question": question,
                "metadata": {"question": question}
            })

    # 2. Baseline tasks (All layers x First Question x Num Baselines)
    if RUN_BASELINE_COMPARISON:
        print("Preparing baseline comparison tasks...")
        for layer_idx in layers_to_interpret:
            vector = torch.from_numpy(directions[layer_idx]).float()
            target_norm = vector.norm()
            
            # Comparison only uses the first question to save time
            question = INTERPRETATION_QUESTIONS[0]
            prompt = f"Layer {layer_idx}:{PLACEHOLDER_TOKEN} {question}"
            
            for i in range(NUM_BASELINE_VECTORS):
                # Generate random vector with same norm
                random_vec = torch.randn_like(vector)
                random_vec = random_vec / random_vec.norm() * target_norm
                
                tasks.append({
                    "type": "baseline",
                    "layer": layer_idx,
                    "vector": random_vec,
                    "prompt": prompt,
                    "question": question,
                    "metadata": {"baseline_idx": i}
                })

    # =========================================================================
    # EXECUTION LOOP
    # =========================================================================

    results = {} # Structure: {layer: {question: result_dict}}
    baseline_temp = {} # Structure: {layer: {baseline_logprobs: [], probe_logprob: float}}

    # Initialize results structure
    for l in layers_to_interpret:
        results[l] = {}
        if RUN_BASELINE_COMPARISON:
            baseline_temp[l] = {"baseline_logprobs": [], "baseline_responses": [], "probe_logprob": 0.0}

    # Helper for batching
    def chunked(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i : i + n]

    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with tqdm(total=len(tasks), desc="Processing Batches") as pbar:
        for batch_tasks in chunked(tasks, BATCH_SIZE):
            
            # Run inference
            batch_results = interpreter.interpret_batch(
                batch_tasks, 
                with_confidence=RUN_CONFIDENCE_TESTS
            )
            
            # Map results back to structure
            for task, res in zip(batch_tasks, batch_results):
                layer = task['layer']
                
                if task['type'] == 'main':
                    q = task['question']

                    # Store in main results (including entropy metrics)
                    results[layer][q] = {
                        "response": res["response"],
                        "mean_logprob": res.get("mean_logprob", 0),
                        "min_logprob": res.get("min_logprob", 0),
                        "mean_entropy": res.get("mean_entropy"),
                        "max_entropy": res.get("max_entropy"),
                        "entropy_trend": res.get("entropy_trend"),
                        "mean_top_k_concentration": res.get("mean_top_k_concentration"),
                    }

                    # If this is the primary question, store for baseline comparison
                    if RUN_BASELINE_COMPARISON and q == INTERPRETATION_QUESTIONS[0]:
                        baseline_temp[layer]["probe_logprob"] = res.get("mean_logprob", 0)
                        
                elif task['type'] == 'baseline':
                    # Store baseline logprob and response
                    baseline_temp[layer]["baseline_logprobs"].append(res.get("mean_logprob", 0))
                    baseline_temp[layer]["baseline_responses"].append(res["response"])

            pbar.update(len(batch_tasks))

    # =========================================================================
    # POST-PROCESSING
    # =========================================================================

    # Finalize baseline results
    baseline_results = {}
    if RUN_BASELINE_COMPARISON:
        for layer_idx, data in baseline_temp.items():
            baseline_logprobs = data["baseline_logprobs"]
            baseline_responses = data["baseline_responses"]
            probe_logprob = data["probe_logprob"]
            baseline_mean = sum(baseline_logprobs) / len(baseline_logprobs) if baseline_logprobs else 0

            baseline_results[layer_idx] = {
                "probe_logprob": probe_logprob,
                "baseline_mean_logprob": baseline_mean,
                "logprob_delta": probe_logprob - baseline_mean,
                "baseline_logprobs": baseline_logprobs,
                "baseline_results": baseline_responses,
            }

    # Analyze results
    print("\nAnalyzing AO outputs...")
    analysis = analyze_ao_outputs(
        results,
        baseline_results if RUN_BASELINE_COMPARISON else None,
    )

    # Print minimal console summary
    print("\n" + "=" * 70)
    print("AO INTERPRETATION SUMMARY")
    print("=" * 70)

    summary = analysis.get("summary", {})
    qc = summary.get("quality_counts", {})
    total = qc.get("total_responses", 0)
    print(f"\nResponse Quality: {qc.get('high_quality', 0)}/{total} high-quality, "
          f"{qc.get('garbage', 0)} garbage, {qc.get('ambiguous', 0)} ambiguous")

    # Top layers per concept
    top_layers = summary.get("top_layers_by_concept", {})
    print(f"\nTop Layers by Concept:")
    for concept, layers in top_layers.items():
        layer_str = ", ".join(str(l) for l in layers[:5])
        print(f"  {concept.upper()}: {layer_str}")

    # Save results
    output_prefix = str(directions_path).replace("_directions.npz", "").replace("_probe_directions.npz", "")
    
    json_path = f"{output_prefix}_ao_interpretations.json"
    results_serializable = {str(k): v for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")

    analysis_path = f"{output_prefix}_ao_analysis.json"
    analysis_serializable = {
        "summary": analysis.get("summary", {}),  # At-a-glance summary first
        "quality_counts": analysis["quality_counts"],
        "concept_signals": analysis["concept_signals"],
        "layer_semantic": {str(k): v for k, v in analysis.get("layer_semantic", {}).items()},
        "layer_quality": {str(k): v for k, v in analysis["layer_quality"].items()},
        "best_layers": [(int(k), float(v)) for k, v in analysis["best_layers"]],
        "question_quality": analysis["question_quality"],
        "per_layer": {str(k): v for k, v in analysis["per_layer"].items()},
    }
    if "layer_confidence" in analysis:
        analysis_serializable["layer_confidence"] = {str(k): v for k, v in analysis["layer_confidence"].items()}
        analysis_serializable["question_confidence"] = analysis["question_confidence"]
    if "layer_entropy" in analysis:
        analysis_serializable["layer_entropy"] = {str(k): v for k, v in analysis["layer_entropy"].items()}
    if "baseline_summary" in analysis:
        analysis_serializable["baseline_summary"] = {str(k): v for k, v in analysis["baseline_summary"].items()}
    # Save baseline text responses if available
    if RUN_BASELINE_COMPARISON and baseline_results:
        analysis_serializable["baseline_responses"] = {
            str(k): v.get("baseline_results", []) for k, v in baseline_results.items()
        }
    with open(analysis_path, "w") as f:
        json.dump(analysis_serializable, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")

    # Save text summary
    summary_path = f"{output_prefix}_ao_summary.txt"
    generate_text_summary(analysis, summary_path)
    print(f"Summary saved to: {summary_path}")

    # Create visualization
    png_path = f"{output_prefix}_ao_visualization.png"
    visualize_ao_results(results, analysis, png_path)

    print(f"\n{'=' * 70}")
    print("INTERPRETATION COMPLETE")
    print("=" * 70)


def analyze_existing_results(json_path: str):
    """
    Analyze and visualize existing AO interpretation results.
    """
    print(f"Loading existing results from: {json_path}")

    with open(json_path) as f:
        results_raw = json.load(f)

    # Convert string keys back to int
    results = {int(k): v for k, v in results_raw.items()}

    print(f"Loaded results for {len(results)} layers")

    # Analyze
    print("\nAnalyzing AO outputs...")
    analysis = analyze_ao_outputs(results)

    # Print minimal console summary
    print("\n" + "=" * 70)
    print("AO INTERPRETATION SUMMARY")
    print("=" * 70)

    summary = analysis.get("summary", {})
    qc = summary.get("quality_counts", {})
    total = qc.get("total_responses", 0)
    print(f"\nResponse Quality: {qc.get('high_quality', 0)}/{total} high-quality, "
          f"{qc.get('garbage', 0)} garbage, {qc.get('ambiguous', 0)} ambiguous")

    # Top layers per concept
    top_layers = summary.get("top_layers_by_concept", {})
    print(f"\nTop Layers by Concept:")
    for concept, layers in top_layers.items():
        layer_str = ", ".join(str(l) for l in layers[:5])
        print(f"  {concept.upper()}: {layer_str}")

    # Save analysis
    output_prefix = json_path.replace("_ao_interpretations.json", "")
    analysis_path = f"{output_prefix}_ao_analysis.json"

    analysis_serializable = {
        "summary": analysis.get("summary", {}),
        "quality_counts": analysis["quality_counts"],
        "concept_signals": analysis["concept_signals"],
        "layer_semantic": {str(k): v for k, v in analysis.get("layer_semantic", {}).items()},
        "layer_quality": {str(k): v for k, v in analysis["layer_quality"].items()},
        "best_layers": [(int(k), float(v)) for k, v in analysis["best_layers"]],
        "question_quality": analysis["question_quality"],
        "per_layer": {str(k): v for k, v in analysis["per_layer"].items()},
    }
    if "layer_confidence" in analysis:
        analysis_serializable["layer_confidence"] = {str(k): v for k, v in analysis["layer_confidence"].items()}
    if "layer_entropy" in analysis:
        analysis_serializable["layer_entropy"] = {str(k): v for k, v in analysis["layer_entropy"].items()}
    with open(analysis_path, "w") as f:
        json.dump(analysis_serializable, f, indent=2)
    print(f"\nAnalysis saved to: {analysis_path}")

    # Save text summary
    summary_path = f"{output_prefix}_ao_summary.txt"
    generate_text_summary(analysis, summary_path)
    print(f"Summary saved to: {summary_path}")

    # Create visualization
    png_path = f"{output_prefix}_ao_visualization.png"
    visualize_ao_results(results, analysis, png_path)

    print("\nDone!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("Usage: python act_oracles.py --analyze <path_to_ao_interpretations.json>")
            sys.exit(1)
        analyze_existing_results(sys.argv[2])
    else:
        main()