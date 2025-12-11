"""
load_collected_data_v2.py

Utilities for loading and working with collected activation data (v2 format with compressed npz).

Example usage:
    from load_collected_data_v2 import ActivationDataset
    
    # Load MCQ data
    dataset = ActivationDataset("full_run_base/mcq")
    
    # Get all metadata
    metadata = dataset.get_metadata()
    
    # Get activations for specific layer and questions
    layer_15_acts = dataset.get_layer_activations(15, question_indices=[0, 1, 2])
    
    # Get all layers at once
    all_acts = dataset.get_all_layers_activations()  # [32, num_questions, 4096]
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any


class ActivationDataset:
    """Load and access collected activation data (v2 compressed format)."""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to data directory (e.g., "full_run_base/mcq")
        """
        self.data_dir = Path(data_dir)
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"No metadata.json in {data_dir}")
        
        with open(metadata_file) as f:
            self.metadata = json.load(f)
        
        self.num_questions = len(self.metadata)
        
        # Load activations (lazy loading - only when needed)
        self._activations = None
        self._activations_file = self.data_dir / "activations.npz"
        
        if not self._activations_file.exists():
            raise ValueError(f"No activations.npz in {data_dir}")
        
        # Load shape info without loading full array
        with np.load(self._activations_file) as data:
            self.num_layers = data['activations'].shape[0]
            self.hidden_dim = data['activations'].shape[2]
        
        print(f"✓ Loaded dataset from {data_dir}")
        print(f"  Questions: {self.num_questions}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden dim: {self.hidden_dim}")
    
    def _load_activations(self):
        """Lazy load activations into memory."""
        if self._activations is None:
            with np.load(self._activations_file) as data:
                self._activations = data['activations']  # [num_layers, num_questions, 4096]
        return self._activations
    
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
            np.ndarray: Shape [num_questions, 4096]
        """
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {self.num_layers})")
        
        acts = self._load_activations()
        layer_acts = acts[layer]  # [num_questions, 4096]
        
        if question_indices is not None:
            layer_acts = layer_acts[question_indices]
        
        return layer_acts
    
    def get_all_layers_activations(
        self,
        question_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get activations for all layers.
        
        Args:
            question_indices: Optional specific questions
        
        Returns:
            np.ndarray: Shape [num_layers, num_questions, 4096]
        """
        acts = self._load_activations()
        
        if question_indices is not None:
            acts = acts[:, question_indices, :]
        
        return acts
    
    def get_activations_for_probing(
        self,
        layer: int,
        correct_only: Optional[bool] = None,
        incorrect_only: Optional[bool] = None,
        entropy_range: Optional[tuple] = None,
        confidence_range: Optional[tuple] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get activations and labels for probe training.
        
        Returns:
            (activations, labels) where:
                activations: [num_samples, 4096]
                labels: [num_samples] (entropy values or confidence values)
        """
        # Get filtered question indices
        indices = self.get_question_indices(
            correct_only=correct_only,
            incorrect_only=incorrect_only,
            entropy_range=entropy_range,
            confidence_range=confidence_range
        )
        
        # Get activations
        acts = self.get_layer_activations(layer, indices)
        
        # Get labels (entropy)
        labels = self.get_entropy_values(indices)
        
        return acts, labels
    
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
        print(f"Layers: {self.num_layers}")
        print(f"Hidden dim: {self.hidden_dim}")
        
        # File size
        file_size_mb = self._activations_file.stat().st_size / (1024**2)
        print(f"Activation file size: {file_size_mb:.1f} MB")
        
        if self.metadata:
            meta = self.metadata[0]
            
            print(f"\nPrompt type: {meta.get('prompt_type', 'N/A')}")
            print(f"Model: {meta.get('model_name', 'N/A')}")
            print(f"Checkpoint: {meta.get('checkpoint_id', 'N/A')}")
            
            # Response length
            if "response_length" in meta:
                lengths = [m.get("response_length", 0) for m in self.metadata]
                print(f"Response tokens: {np.mean(lengths):.1f} avg, {max(lengths)} max")
            
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
        base_dir: Base model data directory
        finetuned_dir: Finetuned model data directory
        prompt_type: "mcq", "self_conf", or "other_conf"
    
    Returns:
        (base_dataset, finetuned_dataset)
    """
    base_path = Path(base_dir) / prompt_type
    fine_path = Path(finetuned_dir) / prompt_type
    
    base = ActivationDataset(str(base_path))
    finetuned = ActivationDataset(str(fine_path))
    
    # Verify they have same number of questions
    if base.num_questions != finetuned.num_questions:
        print(f"⚠️  Warning: Different number of questions!")
        print(f"   Base: {base.num_questions}, Finetuned: {finetuned.num_questions}")
    
    return base, finetuned


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_collected_data_v2.py <data_dir>")
        sys.exit(1)
    
    dataset = ActivationDataset(sys.argv[1])
    dataset.summary()
    
    # Example: Get activations for layer 15
    print("Loading layer 15 activations...")
    acts = dataset.get_layer_activations(15)
    print(f"Shape: {acts.shape}")
    print(f"dtype: {acts.dtype}")
    
    # Example: Get correct vs incorrect
    correct_indices = dataset.get_question_indices(correct_only=True)
    incorrect_indices = dataset.get_question_indices(incorrect_only=True)
    print(f"\nCorrect questions: {len(correct_indices)}")
    print(f"Incorrect questions: {len(incorrect_indices)}")