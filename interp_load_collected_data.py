"""
load_collected_data.py

Utilities for loading and working with collected activation data.

Example usage:
    from load_collected_data import ActivationDataset
    
    # Load MCQ data
    dataset = ActivationDataset("full_run_base/mcq")
    
    # Get all metadata
    metadata = dataset.get_metadata()
    
    # Get activations for specific layer and questions
    layer_15_acts = dataset.get_layer_activations(15, question_indices=[0, 1, 2])
    
    # Get final token activations (most common use case)
    final_acts = dataset.get_final_token_activations(layer=15)
    
    # Filter by correctness
    correct_indices = dataset.get_question_indices(correct_only=True)
    incorrect_acts = dataset.get_final_token_activations(15, correct_only=False)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from finetune_data_handling import validate_file_exists_and_not_empty


class ActivationDataset:
    """Load and access collected activation data."""
    
    def __init__(self, data_dir: str, prompt_type: Optional[str] = None):
        """
        Args:
            data_dir: Path to data directory. Can be:
                - Top-level directory (e.g., "interp_activations_r_10/") - must specify prompt_type
                - Subdirectory (e.g., "interp_activations_r_10/mcq/")
            prompt_type: If data_dir is top-level, specify "mcq", "self_conf", or "other_conf"
        """
        self.data_dir = Path(data_dir)
        
        # Validate directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Check if this is a top-level directory (has subdirectories like mcq/, self_conf/, etc.)
        # or a subdirectory (has metadata.json directly)
        metadata_file = self.data_dir / "metadata.json"
        
        if not metadata_file.exists() and prompt_type:
            # Top-level directory, navigate to prompt_type subdirectory
            self.data_dir = self.data_dir / prompt_type
            metadata_file = self.data_dir / "metadata.json"
        elif not metadata_file.exists():
            # Try to auto-detect prompt type
            for pt in ["mcq", "self_conf", "other_conf"]:
                test_path = self.data_dir / pt / "metadata.json"
                if test_path.exists():
                    self.data_dir = self.data_dir / pt
                    metadata_file = test_path
                    break
        
        if not metadata_file.exists():
            raise ValueError(f"Metadata file not found in {data_dir}. Expected metadata.json or subdirectory with metadata.json")
        
        validate_file_exists_and_not_empty(metadata_file, "metadata file")
        
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        
        self.num_questions = len(self.metadata)
        self.num_layers = 32
        
        # Check for new format (.npz) or old format (individual .npy files)
        npz_file = self.data_dir / "activations.npz"
        self.use_npz_format = npz_file.exists()
        
        if self.use_npz_format:
            # New format: single .npz file with shape [num_layers, num_questions, 4096]
            npz_data = np.load(npz_file)
            self.activations_all = npz_data["activations"]  # [num_layers, num_questions, 4096]
            self.available_layers = list(range(self.activations_all.shape[0]))
            self.num_layers = len(self.available_layers)
        else:
            # Old format: individual .npy files per layer
            self.available_layers = []
            for i in range(self.num_layers):
                layer_file = self.data_dir / f"activations_layer_{i:02d}.npy"
                if layer_file.exists():
                    self.available_layers.append(i)
            self.activations_all = None
        
        if not self.available_layers:
            raise ValueError(f"No activation files found in {data_dir}")
        
        print(f"✓ Loaded dataset from {self.data_dir}")
        print(f"  Questions: {self.num_questions}")
        print(f"  Layers available: {len(self.available_layers)}")
        print(f"  Format: {'NPZ (final token only)' if self.use_npz_format else 'NPY (full sequence)'}")
    
    def get_metadata(self, question_indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get metadata for questions."""
        if question_indices is None:
            return self.metadata
        return [self.metadata[i] for i in question_indices]
    
    def get_layer_activations(
        self,
        layer: int,
        question_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get activations for a specific layer.
        
        Args:
            layer: Layer index (0-31)
            question_indices: Optional list of question indices to load
        
        Returns:
            np.ndarray: 
                - New format (NPZ): Shape [num_questions, 4096] (final token only)
                - Old format (NPY): Shape [num_questions, seq_length, 4096]
        """
        if layer not in self.available_layers:
            raise ValueError(f"Layer {layer} not available. Available: {self.available_layers}")
        
        if self.use_npz_format:
            # New format: [num_layers, num_questions, 4096]
            acts = self.activations_all[layer]  # [num_questions, 4096]
            if question_indices is not None:
                acts = acts[question_indices]
        else:
            # Old format: load from individual .npy file
            layer_file = self.data_dir / f"activations_layer_{layer:02d}.npy"
            acts = np.load(layer_file)  # [num_questions, seq_length, 4096]
            if question_indices is not None:
                acts = acts[question_indices]
        
        return acts
    
    def get_final_token_activations(
        self,
        layer: int,
        question_indices: Optional[List[int]] = None,
        correct_only: Optional[bool] = None,
        incorrect_only: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Get final token activations for a layer.
        
        Args:
            layer: Layer index
            question_indices: Optional specific questions
            correct_only: Only return activations for correct answers
            incorrect_only: Only return activations for incorrect answers
        
        Returns:
            np.ndarray: Shape [num_questions, 4096]
        """
        # Apply correctness filter
        if correct_only or incorrect_only:
            indices = self.get_question_indices(
                correct_only=correct_only,
                incorrect_only=incorrect_only
            )
            if question_indices is not None:
                # Intersection
                indices = [i for i in indices if i in question_indices]
        else:
            indices = question_indices
        
        # Load activations
        acts = self.get_layer_activations(layer, indices)
        
        # For new format (NPZ), activations are already final token only [num_questions, 4096]
        if self.use_npz_format:
            return acts
        
        # For old format (NPY), extract final tokens from [num_questions, seq_length, 4096]
        final_acts = []
        metadata_subset = self.get_metadata(indices)
        
        for i, meta in enumerate(metadata_subset):
            # Get token index for this question
            if "answer_token_index" in meta:
                token_idx = meta["answer_token_index"]
            elif "confidence_token_index" in meta:
                token_idx = meta["confidence_token_index"]
            else:
                # Fallback: use last token
                token_idx = -1
            
            final_acts.append(acts[i, token_idx])
        
        return np.array(final_acts)
    
    def get_all_layers_final_token(
        self,
        question_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[int, np.ndarray]:
        """
        Get final token activations for all layers.
        
        Returns:
            Dict mapping layer_idx -> activations [num_questions, 4096]
        """
        result = {}
        for layer in self.available_layers:
            result[layer] = self.get_final_token_activations(
                layer, question_indices, **kwargs
            )
        return result
    
    def get_question_indices(
        self,
        correct_only: bool = False,
        incorrect_only: bool = False,
        entropy_range: Optional[tuple] = None,
        confidence_range: Optional[tuple] = None,
    ) -> List[int]:
        """
        Get question indices matching criteria.
        
        Args:
            correct_only: Only correct answers (MCQ only)
            incorrect_only: Only incorrect answers (MCQ only)
            entropy_range: (min, max) entropy range
            confidence_range: (min, max) confidence range
        
        Returns:
            List of question indices
        """
        indices = []
        
        for i, meta in enumerate(self.metadata):
            # Correctness filter
            if correct_only and not meta.get("is_correct", False):
                continue
            if incorrect_only and meta.get("is_correct", True):
                continue
            
            # Entropy filter
            if entropy_range is not None:
                entropy = meta.get("entropy", float("inf"))
                if not (entropy_range[0] <= entropy <= entropy_range[1]):
                    continue
            
            # Confidence filter
            if confidence_range is not None:
                conf = meta.get("self_confidence") or meta.get("other_confidence", -1)
                if not (confidence_range[0] <= conf <= confidence_range[1]):
                    continue
            
            indices.append(i)
        
        return indices
    
    def get_entropy_values(self, question_indices: Optional[List[int]] = None) -> np.ndarray:
        """Get entropy values for questions."""
        metadata = self.get_metadata(question_indices)
        return np.array([m["entropy"] for m in metadata])
    
    def get_confidence_values(
        self,
        question_indices: Optional[List[int]] = None,
        confidence_type: str = "self"
    ) -> np.ndarray:
        """
        Get confidence values.
        
        Args:
            confidence_type: "self" or "other"
        """
        metadata = self.get_metadata(question_indices)
        key = f"{confidence_type}_confidence"
        return np.array([m[key] for m in metadata if key in m])
    
    def get_surface_features(
        self,
        question_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get surface features as arrays.
        
        Returns:
            Dict mapping feature_name -> values
        """
        metadata = self.get_metadata(question_indices)
        
        if not metadata or "surface_features" not in metadata[0]:
            return {}
        
        # Get all feature names
        feature_names = metadata[0]["surface_features"].keys()
        
        result = {}
        for name in feature_names:
            values = [m["surface_features"][name] for m in metadata]
            
            # Convert to numeric if possible
            if isinstance(values[0], (int, float)):
                result[name] = np.array(values)
            else:
                result[name] = values  # Keep as list for strings/bools
        
        return result
    
    def summary(self):
        """Print dataset summary."""
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY: {self.data_dir}")
        print(f"{'='*60}")
        print(f"Questions: {self.num_questions}")
        print(f"Layers: {len(self.available_layers)}")
        
        if self.metadata:
            meta = self.metadata[0]
            
            print(f"\nPrompt type: {meta.get('prompt_type', 'N/A')}")
            print(f"Model: {meta.get('model_name', 'N/A')}")
            print(f"Checkpoint: {meta.get('checkpoint_id', 'N/A')}")
            
            # Accuracy
            if "is_correct" in meta:
                correct = sum(1 for m in self.metadata if m.get("is_correct", False))
                acc = correct / self.num_questions * 100
                print(f"\nAccuracy: {correct}/{self.num_questions} ({acc:.1f}%)")
            
            # Entropy
            if "entropy" in meta:
                entropies = self.get_entropy_values()
                print(f"\nEntropy: mean={entropies.mean():.3f}, std={entropies.std():.3f}")
            
            # Confidence
            if "self_confidence" in meta:
                confs = self.get_confidence_values("self")
                print(f"Self-confidence: mean={confs.mean():.1f}%, std={confs.std():.1f}%")
            
            if "other_confidence" in meta:
                confs = self.get_confidence_values("other")
                print(f"Other-confidence: mean={confs.mean():.1f}%, std={confs.std():.1f}%")
        
        print(f"{'='*60}\n")


def load_paired_datasets(base_dir: str, finetuned_dir: str, prompt_type: str = "mcq"):
    """
    Load base and finetuned datasets for comparison.
    
    Args:
        base_dir: Base model data directory (can be top-level or subdirectory)
        finetuned_dir: Finetuned model data directory (can be top-level or subdirectory)
        prompt_type: "mcq", "self_conf", or "other_conf" (only needed if directories are top-level)
    
    Returns:
        (base_dataset, finetuned_dataset)
    """
    base = ActivationDataset(base_dir, prompt_type=prompt_type)
    finetuned = ActivationDataset(finetuned_dir, prompt_type=prompt_type)
    
    # Verify they have same number of questions
    if base.num_questions != finetuned.num_questions:
        print(f"⚠️  Warning: Different number of questions!")
        print(f"   Base: {base.num_questions}, Finetuned: {finetuned.num_questions}")
    
    return base, finetuned


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_collected_data.py <data_dir>")
        sys.exit(1)
    
    dataset = ActivationDataset(sys.argv[1])
    dataset.summary()
    
    # Example: Get final token activations for layer 15
    print("Loading layer 15 final token activations...")
    acts = dataset.get_final_token_activations(15)
    print(f"Shape: {acts.shape}")
    print(f"dtype: {acts.dtype}")
    
    # Example: Get correct vs incorrect
    correct_indices = dataset.get_question_indices(correct_only=True)
    incorrect_indices = dataset.get_question_indices(incorrect_only=True)
    print(f"\nCorrect questions: {len(correct_indices)}")
    print(f"Incorrect questions: {len(incorrect_indices)}")
