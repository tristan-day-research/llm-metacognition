"""Pre-flight checks — run before heavy imports to fail fast on missing credentials."""
import os


def check_hf_login():
    """Abort immediately if not logged in to Hugging Face."""
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return  # will surface a clearer error when transformers tries to load
    if (
        HfFolder.get_token() is None
        and not os.environ.get("HF_TOKEN")
        and not os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        raise SystemExit(
            "\n✗  Not logged in to Hugging Face.\n"
            "   Run:  huggingface-cli login\n"
            "   or set the HF_TOKEN environment variable.\n"
        )
