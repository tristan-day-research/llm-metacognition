"""
push_checkpoint.py — push a single local checkpoint dir to HuggingFace Hub.

Workflow this script supports:
  1. Train with SAVE_HF_CHECKPOINTS=False, KEEP_LOCAL_CHECKPOINTS=True so all
     checkpoints stay on disk and nothing gets uploaded automatically.
  2. After training, decide which (if any) checkpoint(s) you want on the Hub.
  3. Run this script to push the chosen one(s); optionally delete the rest.

Usage:
  # Push one checkpoint to <username>/<auto-derived-name>:
  python finetune/push_checkpoint.py PATH_TO_CKPT_DIR --username Tristan-Day

  # Push to an explicit repo name (overrides auto-naming):
  python finetune/push_checkpoint.py PATH_TO_CKPT_DIR --repo Tristan-Day/my-run

  # Make it private:
  python finetune/push_checkpoint.py PATH_TO_CKPT_DIR --username Tristan-Day --private

  # Delete the local copy after a successful push:
  python finetune/push_checkpoint.py PATH_TO_CKPT_DIR --username Tristan-Day --delete-local
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import shutil
from pathlib import Path

from utils import _sanitize_for_hf


def _derive_repo_name(ckpt_path: Path, username: str) -> str:
    """Repo name = <username>/<sanitized parent dir>_<sanitized ckpt dir>.

    The parent directory is the wandb-named checkpoint base dir; the leaf
    is e.g. ``ckpt_step_2000``. Combining them keeps the run identifier
    while disambiguating between steps from the same run.
    """
    parent_slug = _sanitize_for_hf(ckpt_path.parent.name)
    leaf_slug = _sanitize_for_hf(ckpt_path.name)
    base_slug = f"{parent_slug}_{leaf_slug}".strip("-.")
    # HF repo names are capped at 96 chars; keep some headroom for the prefix.
    max_base = 96 - len(username) - 1
    if len(base_slug) > max_base:
        base_slug = base_slug[:max_base].rstrip("-.")
    return f"{username}/{base_slug}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_dir", help="Path to the local checkpoint directory")
    parser.add_argument("--repo", default=None,
                        help="Full HF repo path 'username/name'. If not set, --username is required.")
    parser.add_argument("--username", default=None,
                        help="HF username; the repo name is derived from the checkpoint dir.")
    parser.add_argument("--private", action="store_true", help="Make the HF repo private.")
    parser.add_argument("--delete-local", action="store_true",
                        help="Delete the local checkpoint dir after a successful push.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_dir).resolve()
    if not ckpt_path.is_dir():
        raise SystemExit(f"✗  Not a directory: {ckpt_path}")

    if args.repo:
        repo = args.repo
    elif args.username:
        repo = _derive_repo_name(ckpt_path, args.username)
    else:
        raise SystemExit("✗  Provide either --repo or --username.")

    # Lightweight HF login check (lifted from interp_experiments/preflight.py).
    try:
        from huggingface_hub import HfFolder
        if HfFolder.get_token() is None and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            raise SystemExit(
                "\n✗  Not logged in to Hugging Face.\n"
                "   Run:  huggingface-cli login\n"
            )
    except ImportError:
        pass

    # Heavy imports last so login failures fail fast.
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM

    print(f"Loading checkpoint from: {ckpt_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(str(ckpt_path))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))

    print(f"Pushing to: {repo}  (private={args.private})")
    model.push_to_hub(repo, private=args.private)
    tokenizer.push_to_hub(repo, private=args.private)
    print(f"✓ Pushed to https://huggingface.co/{repo}")

    if args.delete_local:
        shutil.rmtree(ckpt_path)
        print(f"✓ Deleted local: {ckpt_path}")


if __name__ == "__main__":
    main()
