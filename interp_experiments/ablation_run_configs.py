"""
Shared run-config list for the stated-confidence ablation experiment.

The 6 runs cover {base, instruct, finetuned} x {SimpleMC, TriviaMC}, all using
the 1-10 numeric confidence scale (the letter scale works poorly for the base
model). Every run points at an existing introspection output folder so we can
reuse its direct-pass activations and stated_confidence_numeric values without
a new MC forward pass.

Both build_mc_inputs_from_introspection.py and run_all_stated_conf_ablations.py
import this list so they stay in lockstep.
"""

# --- repo path bootstrap (so `from core import ...` works when this file is
# run from anywhere or as `python experiments/run_xxx.py` from the repo root) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))


from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from core.model_utils import get_model_short_name


REPO_ROOT = Path(__file__).parent
OUTPUT_DIR = REPO_ROOT / "outputs"

BASE_MODEL = "meta-llama/Llama-3.1-8B"
INSTRUCT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FINETUNE_ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"

FINETUNE_STEM_PREFIX = (
    "Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000"
)

# Suffix appended to the ablation base_name so 1-10 scale artifacts don't
# collide with any letter-scale runs. run_ablation_causality.py will load
# {base_name}_mc_stated_confidence_directions.npz for this suffixed name.
SCALE_SUFFIX = "scale-numeric"


@dataclass
class RunConfig:
    label: str
    base_model: str
    adapter: Optional[str]
    dataset: str
    intro_dir: Path
    intro_stem: str  # prefix before "_direct_activations.npz" / "_paired_data.json"

    @property
    def base_name(self) -> str:
        """Matches identify_mc_correlate.py:243-247, plus a scale suffix.

        Used as INPUT_BASE_NAME in run_ablation_causality.py — all its input
        and output filenames are built from this.
        """
        model_short = get_model_short_name(self.base_model)
        if self.adapter:
            core = f"{model_short}_adapter-{get_model_short_name(self.adapter)}_{self.dataset}"
        else:
            core = f"{model_short}_{self.dataset}"
        return f"{core}_{SCALE_SUFFIX}"

    @property
    def direct_activations_path(self) -> Path:
        return self.intro_dir / f"{self.intro_stem}_direct_activations.npz"

    @property
    def paired_data_path(self) -> Path:
        return self.intro_dir / f"{self.intro_stem}_paired_data.json"


# run_introspection_for_ablation.py writes all artifacts to outputs/ directly,
# so every RunConfig points at OUTPUT_DIR rather than a model-specific subfolder.
RUNS: List[RunConfig] = [
    RunConfig(
        label="base_SimpleMC",
        base_model=BASE_MODEL, adapter=None, dataset="SimpleMC",
        intro_dir=OUTPUT_DIR,
        intro_stem="Llama-3.1-8B_SimpleMC_introspection_scale-numeric",
    ),
    RunConfig(
        label="base_TriviaMC",
        base_model=BASE_MODEL, adapter=None, dataset="TriviaMC",
        intro_dir=OUTPUT_DIR,
        intro_stem="Llama-3.1-8B_TriviaMC_introspection_scale-numeric",
    ),
    RunConfig(
        label="instruct_SimpleMC",
        base_model=INSTRUCT_MODEL, adapter=None, dataset="SimpleMC",
        intro_dir=OUTPUT_DIR,
        intro_stem="Llama-3.1-8B-Instruct_SimpleMC_introspection_scale-numeric",
    ),
    RunConfig(
        label="instruct_TriviaMC",
        base_model=INSTRUCT_MODEL, adapter=None, dataset="TriviaMC",
        intro_dir=OUTPUT_DIR,
        intro_stem="Llama-3.1-8B-Instruct_TriviaMC_introspection_scale-numeric",
    ),
    RunConfig(
        label="finetuned_SimpleMC",
        base_model=INSTRUCT_MODEL, adapter=FINETUNE_ADAPTER, dataset="SimpleMC",
        intro_dir=OUTPUT_DIR,
        intro_stem=f"{FINETUNE_STEM_PREFIX}_SimpleMC_introspection_scale-numeric",
    ),
    RunConfig(
        label="finetuned_TriviaMC",
        base_model=INSTRUCT_MODEL, adapter=FINETUNE_ADAPTER, dataset="TriviaMC",
        intro_dir=OUTPUT_DIR,
        intro_stem=f"{FINETUNE_STEM_PREFIX}_TriviaMC_introspection_scale-numeric",
    ),
]
