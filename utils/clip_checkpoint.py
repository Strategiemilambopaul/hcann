"""Checkpoint CLIP local, pour éviter un téléchargement Hugging Face au banc."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_LOCAL = os.path.join(REPO_ROOT, "checkpoints", "clip-vit-base-patch32")


def resolve_clip_model(use_clip: bool) -> str | None:
    weights = os.path.join(CLIP_LOCAL, "pytorch_model.bin")
    if use_clip and os.path.isfile(weights):
        return CLIP_LOCAL
    return None


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None) or ""
        if enc.lower() not in ("utf-8", "utf8"):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass
