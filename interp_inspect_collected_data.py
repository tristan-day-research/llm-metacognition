"""
inspect_collected_data.py

Quick inspection script to verify collected data and show summary statistics.

Usage:
    python inspect_collected_data.py --data_dir activations_data/mcq
    python inspect_collected_data.py --data_dir test_run --show_examples 3
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter
from finetune_data_handling import validate_file_exists_and_not_empty


def inspect_data(data_dir, show_examples=1, prompt_type: Optional[str] = None):
    """Inspect collected activation data.
    
    Args:
        data_dir: Path to data directory (can be top-level or subdirectory)
        show_examples: Number of examples to show
        prompt_type: If data_dir is top-level, specify "mcq", "self_conf", or "other_conf"
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return
    
    # Check if this is a top-level directory or subdirectory
    metadata_file = data_path / "metadata.json"
    
    if not metadata_file.exists() and prompt_type:
        # Top-level directory, navigate to prompt_type subdirectory
        data_path = data_path / prompt_type
        metadata_file = data_path / "metadata.json"
    elif not metadata_file.exists():
        # Try to auto-detect prompt type
        for pt in ["mcq", "self_conf", "other_conf"]:
            test_path = data_path / pt / "metadata.json"
            if test_path.exists():
                data_path = data_path / pt
                metadata_file = test_path
                break
    
    print(f"\n{'='*60}")
    print(f"DATA INSPECTION: {data_path}")
    print(f"{'='*60}\n")
    
    # Load metadata
    try:
        validate_file_exists_and_not_empty(metadata_file, "metadata file")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ {e}")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"📊 Number of questions: {len(metadata)}")
    
    # Check activation files (new format: .npz or old format: individual .npy)
    npz_file = data_path / "activations.npz"
    layer_files = sorted(data_path.glob("activations_layer_*.npy"))
    
    if npz_file.exists():
        # New format: single .npz file
        print(f"📦 Activation format: NPZ (compressed, final token only)")
        npz_data = np.load(npz_file)
        activations = npz_data["activations"]  # [num_layers, num_questions, 4096]
        print(f"📐 Activation shape: {activations.shape}")
        print(f"   [num_layers, num_questions, hidden_dim]")
        print(f"   dtype: {activations.dtype}")
        
        file_size_mb = npz_file.stat().st_size / (1024**2)
        print(f"💾 Total activation storage: {file_size_mb:.1f} MB (compressed)")
        
        # Calculate uncompressed size estimate
        uncompressed_mb = activations.nbytes / (1024**2)
        compression_ratio = uncompressed_mb / file_size_mb if file_size_mb > 0 else 1.0
        print(f"   Uncompressed size: ~{uncompressed_mb:.1f} MB")
        print(f"   Compression ratio: ~{compression_ratio:.1f}x")
    elif layer_files:
        # Old format: individual .npy files
        print(f"📦 Activation format: NPY (individual files, full sequence)")
        print(f"📦 Activation layers found: {len(layer_files)}")
        
        # Load one layer to check shape
        sample_layer = np.load(layer_files[0])
        print(f"📐 Activation shape per layer: {sample_layer.shape}")
        print(f"   [num_questions, seq_length, hidden_dim]")
        print(f"   dtype: {sample_layer.dtype}")
        
        # Calculate total size
        total_size_mb = sum(f.stat().st_size for f in layer_files) / (1024**2)
        print(f"💾 Total activation storage: {total_size_mb:.1f} MB")
    else:
        print(f"⚠️  No activation files found (expected activations.npz or activations_layer_*.npy)")
    
    # Check for duplicate questions
    print(f"\n{'='*60}")
    print("DUPLICATE QUESTION CHECK")
    print(f"{'='*60}\n")
    
    question_ids = [m.get("question_id") for m in metadata if "question_id" in m]
    question_texts = [m.get("question_text") for m in metadata if "question_text" in m]
    
    # Check for duplicate IDs
    id_counts = Counter(question_ids)
    duplicate_ids = {qid: count for qid, count in id_counts.items() if count > 1}
    
    # Check for duplicate question texts
    text_counts = Counter(question_texts)
    duplicate_texts = {text: count for text, count in text_counts.items() if count > 1}
    
    if duplicate_ids:
        print(f"⚠️  Found {len(duplicate_ids)} question IDs that appear multiple times:")
        for qid, count in list(duplicate_ids.items())[:10]:  # Show first 10
            print(f"   '{qid}': {count} times")
        if len(duplicate_ids) > 10:
            print(f"   ... and {len(duplicate_ids) - 10} more")
    else:
        print(f"✓ No duplicate question IDs found")
    
    if duplicate_texts:
        print(f"\n⚠️  Found {len(duplicate_texts)} question texts that appear multiple times:")
        for text, count in list(duplicate_texts.items())[:5]:  # Show first 5
            print(f"   '{text[:60]}...': {count} times")
        if len(duplicate_texts) > 5:
            print(f"   ... and {len(duplicate_texts) - 5} more")
    else:
        print(f"✓ No duplicate question texts found")
    
    # Metadata statistics
    print(f"\n{'='*60}")
    print("METADATA STATISTICS")
    print(f"{'='*60}\n")
    
    if metadata:
        first = metadata[0]
        
        print("Available fields:")
        for key in sorted(first.keys()):
            print(f"  • {key}")
        
        # Prompt type
        if "prompt_type" in first:
            print(f"\nPrompt type: {first['prompt_type']}")
        
        # Model info
        if "model_name" in first:
            print(f"Model: {first['model_name']}")
        if "checkpoint_id" in first:
            print(f"Checkpoint: {first['checkpoint_id']}")
        
        # Accuracy (if MCQ)
        if "is_correct" in first:
            correct = sum(1 for m in metadata if m.get("is_correct", False))
            accuracy = correct / len(metadata) * 100
            print(f"\n✓ Accuracy: {correct}/{len(metadata)} ({accuracy:.1f}%)")
            
            # Distribution of correct answer letters and model predictions
            correct_letters = [m.get("correct_answer_letter") for m in metadata if "correct_answer_letter" in m]
            pred_letters = [m.get("parsed_answer") for m in metadata if "parsed_answer" in m]
            
            if correct_letters:
                correct_dist = Counter(correct_letters)
                print(f"\n📊 Correct Answer Distribution:")
                for letter in "ABCD":
                    count = correct_dist.get(letter, 0)
                    pct = count / len(correct_letters) * 100
                    print(f"   {letter}: {count:4d} ({pct:5.1f}%)")
            
            if pred_letters:
                pred_dist = Counter(pred_letters)
                print(f"\n📊 Model Prediction Distribution:")
                for letter in "ABCD":
                    count = pred_dist.get(letter, 0)
                    pct = count / len(pred_letters) * 100
                    print(f"   {letter}: {count:4d} ({pct:5.1f}%)")
        
        # Entropy statistics
        if "entropy" in first:
            entropies = [m["entropy"] for m in metadata]
            print(f"\n📉 Entropy statistics:")
            print(f"   Mean: {np.mean(entropies):.3f}")
            print(f"   Std:  {np.std(entropies):.3f}")
            print(f"   Min:  {np.min(entropies):.3f}")
            print(f"   Max:  {np.max(entropies):.3f}")
        
        # Confidence (if confidence task)
        if "self_confidence" in first:
            confs = [m["self_confidence"] for m in metadata]
            print(f"\n🎯 Self-confidence statistics:")
            print(f"   Mean: {np.mean(confs):.1f}%")
            print(f"   Std:  {np.std(confs):.1f}%")
            print(f"   Min:  {np.min(confs):.1f}%")
            print(f"   Max:  {np.max(confs):.1f}%")
            
            # Distribution of self-confidence bin predictions
            pred_bins = [m.get("parsed_answer") for m in metadata if "parsed_answer" in m]
            if pred_bins:
                bin_dist = Counter(pred_bins)
                print(f"\n📊 Self-Confidence Bin Distribution:")
                bin_labels = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", 
                             "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
                for letter in "ABCDEFGH":
                    count = bin_dist.get(letter, 0)
                    pct = count / len(pred_bins) * 100
                    label = bin_labels.get(letter, letter)
                    print(f"   {letter} ({label:>6s}): {count:4d} ({pct:5.1f}%)")
        
        if "other_confidence" in first:
            confs = [m["other_confidence"] for m in metadata]
            print(f"\n👥 Other-confidence statistics:")
            print(f"   Mean: {np.mean(confs):.1f}%")
            print(f"   Std:  {np.std(confs):.1f}%")
            print(f"   Min:  {np.min(confs):.1f}%")
            print(f"   Max:  {np.max(confs):.1f}%")
            
            # Distribution of other-confidence bin predictions
            pred_bins = [m.get("parsed_answer") for m in metadata if "parsed_answer" in m]
            if pred_bins:
                bin_dist = Counter(pred_bins)
                print(f"\n📊 Other-Confidence Bin Distribution:")
                bin_labels = {"A": "<5%", "B": "5-10%", "C": "10-20%", "D": "20-40%", 
                             "E": "40-60%", "F": "60-80%", "G": "80-90%", "H": ">90%"}
                for letter in "ABCDEFGH":
                    count = bin_dist.get(letter, 0)
                    pct = count / len(pred_bins) * 100
                    label = bin_labels.get(letter, letter)
                    print(f"   {letter} ({label:>6s}): {count:4d} ({pct:5.1f}%)")
        
        # Surface features
        if "surface_features" in first:
            print(f"\n📝 Surface feature availability:")
            features = first["surface_features"]
            for key in sorted(features.keys()):
                print(f"   • {key}: {features[key]}")
    
    # Show examples
    if show_examples > 0 and metadata:
        print(f"\n{'='*60}")
        print(f"EXAMPLE ENTRIES (showing {min(show_examples, len(metadata))})")
        print(f"{'='*60}\n")
        
        for i in range(min(show_examples, len(metadata))):
            m = metadata[i]
            print(f"Question {i+1}:")
            print(f"  ID: {m.get('question_id', 'N/A')}")
            print(f"  Text: {m['question_text'][:80]}...")
            
            if "correct_answer_letter" in m:
                print(f"  Correct: {m['correct_answer_letter']}")
                print(f"  Predicted: {m['parsed_answer']}")
                print(f"  ✓ Correct" if m.get("is_correct") else "  ✗ Incorrect")
            
            if "entropy" in m:
                print(f"  Entropy: {m['entropy']:.3f}")
            
            if "self_confidence" in m:
                print(f"  Self-confidence: {m['self_confidence']:.1f}%")
            
            if "other_confidence" in m:
                print(f"  Other-confidence: {m['other_confidence']:.1f}%")
            
            print()
    
    print(f"{'='*60}\n")


def compare_runs(base_dir, finetuned_dir, prompt_type: Optional[str] = None):
    """Compare base vs finetuned model results.
    
    Args:
        base_dir: Base model data directory (can be top-level or subdirectory)
        finetuned_dir: Finetuned model data directory (can be top-level or subdirectory)
        prompt_type: If directories are top-level, specify "mcq", "self_conf", or "other_conf"
    """
    
    print(f"\n{'='*60}")
    print("COMPARING BASE vs FINETUNED")
    print(f"{'='*60}\n")
    
    # Load both - handle top-level or subdirectory
    base_path = Path(base_dir)
    fine_path = Path(finetuned_dir)
    
    # Find metadata files
    base_meta_file = base_path / "metadata.json"
    if not base_meta_file.exists() and prompt_type:
        base_meta_file = base_path / prompt_type / "metadata.json"
    elif not base_meta_file.exists():
        # Try auto-detect
        for pt in ["mcq", "self_conf", "other_conf"]:
            test_path = base_path / pt / "metadata.json"
            if test_path.exists():
                base_meta_file = test_path
                break
    
    fine_meta_file = fine_path / "metadata.json"
    if not fine_meta_file.exists() and prompt_type:
        fine_meta_file = fine_path / prompt_type / "metadata.json"
    elif not fine_meta_file.exists():
        # Try auto-detect
        for pt in ["mcq", "self_conf", "other_conf"]:
            test_path = fine_path / pt / "metadata.json"
            if test_path.exists():
                fine_meta_file = test_path
                break
    
    with open(base_meta_file) as f:
        base_meta = json.load(f)
    
    with open(fine_meta_file) as f:
        fine_meta = json.load(f)
    
    print(f"Base model: {len(base_meta)} questions")
    print(f"Finetuned model: {len(fine_meta)} questions")
    
    # Compare accuracy (if MCQ)
    if "is_correct" in base_meta[0]:
        base_correct = sum(1 for m in base_meta if m.get("is_correct", False))
        fine_correct = sum(1 for m in fine_meta if m.get("is_correct", False))
        
        base_acc = base_correct / len(base_meta) * 100
        fine_acc = fine_correct / len(fine_meta) * 100
        
        print(f"\n📊 Accuracy:")
        print(f"   Base: {base_acc:.1f}% ({base_correct}/{len(base_meta)})")
        print(f"   Finetuned: {fine_acc:.1f}% ({fine_correct}/{len(fine_meta)})")
        print(f"   Δ Improvement: {fine_acc - base_acc:+.1f}%")
    
    # Compare entropy
    if "entropy" in base_meta[0]:
        base_entropy = np.array([m["entropy"] for m in base_meta])
        fine_entropy = np.array([m["entropy"] for m in fine_meta])
        
        print(f"\n📉 Entropy:")
        print(f"   Base mean: {base_entropy.mean():.3f} (std: {base_entropy.std():.3f})")
        print(f"   Finetuned mean: {fine_entropy.mean():.3f} (std: {fine_entropy.std():.3f})")
        print(f"   Δ Change: {fine_entropy.mean() - base_entropy.mean():.3f}")
    
    # Compare confidence (if available)
    if "self_confidence" in base_meta[0]:
        base_conf = np.array([m["self_confidence"] for m in base_meta])
        fine_conf = np.array([m["self_confidence"] for m in fine_meta])
        
        print(f"\n🎯 Self-Confidence:")
        print(f"   Base mean: {base_conf.mean():.1f}%")
        print(f"   Finetuned mean: {fine_conf.mean():.1f}%")
        print(f"   Δ Change: {fine_conf.mean() - base_conf.mean():+.1f}%")


def check_duplicates_across_types(data_dir):
    """Check for duplicate questions across all prompt types in a top-level directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Directory not found: {data_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"DUPLICATE CHECK ACROSS ALL PROMPT TYPES: {data_path}")
    print(f"{'='*60}\n")
    
    prompt_types = ["mcq", "self_conf", "other_conf"]
    all_question_ids = {}
    all_question_texts = {}
    
    for pt in prompt_types:
        meta_file = data_path / pt / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                metadata = json.load(f)
            
            question_ids = [m.get("question_id") for m in metadata if "question_id" in m]
            question_texts = [m.get("question_text") for m in metadata if "question_text" in m]
            
            all_question_ids[pt] = set(question_ids)
            all_question_texts[pt] = set(question_texts)
            
            print(f"{pt}: {len(question_ids)} questions")
        else:
            print(f"{pt}: No metadata.json found")
    
    # Check for duplicates within each type
    print(f"\nWithin-type duplicates:")
    for pt in prompt_types:
        if pt in all_question_ids:
            ids = list(all_question_ids[pt])
            id_counts = Counter(ids)
            duplicates = {qid: count for qid, count in id_counts.items() if count > 1}
            if duplicates:
                print(f"  {pt}: {len(duplicates)} duplicate IDs")
            else:
                print(f"  {pt}: ✓ No duplicates")
    
    # Check for questions that appear in multiple types
    print(f"\nCross-type overlap:")
    if len(all_question_ids) > 1:
        types_list = list(all_question_ids.keys())
        for i, pt1 in enumerate(types_list):
            for pt2 in types_list[i+1:]:
                overlap = all_question_ids[pt1] & all_question_ids[pt2]
                if overlap:
                    print(f"  {pt1} & {pt2}: {len(overlap)} shared question IDs")
                else:
                    print(f"  {pt1} & {pt2}: ✓ No overlap")
    
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing collected data (can be top-level or subdirectory)")
    parser.add_argument("--show_examples", type=int, default=1, help="Number of example entries to show")
    parser.add_argument("--compare_with", type=str, default=None, help="Compare with another data directory")
    parser.add_argument("--prompt_type", type=str, default=None, 
                       choices=["mcq", "self_conf", "other_conf"],
                       help="Prompt type if data_dir is top-level directory")
    parser.add_argument("--check_all_types", action="store_true",
                       help="Check for duplicates across all prompt types (for top-level directory)")
    
    args = parser.parse_args()
    
    if args.check_all_types:
        check_duplicates_across_types(args.data_dir)
    else:
        inspect_data(args.data_dir, args.show_examples, prompt_type=args.prompt_type)
    
    if args.compare_with:
        compare_runs(args.data_dir, args.compare_with)


if __name__ == "__main__":
    main()
