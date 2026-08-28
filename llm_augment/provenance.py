"""Provenance sidecars for generated feature matrices.

``user_profile_feat.npy`` and ``text_llm_feat.npy`` are plain arrays: nothing
inside them records which model wrote the text, at what temperature, from which
prompt revision, or whether the backend was the real LLM or the ``echo`` stub.
Two files with identical names and shapes could therefore come from completely
different pipelines, and the only way to tell was to trust that whoever ran the
job remembered what they passed on the command line.

Every build now writes ``<artifact>.meta.json`` next to the array, recording the
generation settings, a content hash of the matrix, and the validation tally.
:func:`describe_artifact` renders it for logs, and the training pipeline reads
the hash to key its caches on *content* rather than on the filename.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Bump when a prompt in prompts.py changes in a way that alters the generated
# text. Two artifacts with different prompt versions are not comparable.
PROMPT_VERSION = "2026-08-01.a"

META_SUFFIX = ".meta.json"


def array_hash(array: np.ndarray) -> str:
    """Short, stable content hash of a feature matrix.

    Hashes shape and dtype alongside the bytes so two arrays that share raw
    bytes but differ in interpretation do not collide.
    """
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def _git_commit() -> Optional[str]:
    """Current commit of the checkout, or ``None`` outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def meta_path_for(artifact_path: str | Path) -> Path:
    """Sidecar path for an artifact (``x.npy`` -> ``x.npy.meta.json``)."""
    return Path(str(artifact_path) + META_SUFFIX)


def write_provenance(
    artifact_path: str | Path,
    array: np.ndarray,
    *,
    kind: str,
    backend: str,
    llm_model: str,
    encoder: str,
    temperature: float,
    max_new_tokens: int,
    n_generated: int,
    train_filtered: bool,
    validation: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write ``<artifact>.meta.json`` and return its path.

    Args:
        artifact_path: the ``.npy`` that was just saved.
        array: the saved matrix, hashed for the cache key.
        kind: ``"user_profile"`` or ``"item_text"``.
        backend: ``hf`` / ``vllm`` / ``echo``.
        train_filtered: whether generation saw only train-split interactions.
            ``False`` means the artifact may contain held-out signal and is not
            valid for reported results.
        validation: the :class:`~llm_augment.validation.ValidationStats` tally.
    """
    meta: dict[str, Any] = {
        "artifact": Path(artifact_path).name,
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "content_hash": array_hash(array),
        "generation": {
            "backend": backend,
            "llm_model": llm_model if backend != "echo" else None,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "prompt_version": PROMPT_VERSION,
            "n_generated": n_generated,
        },
        "encoder": encoder,
        "train_filtered": bool(train_filtered),
        "is_dry_run": backend == "echo",
        "git_commit": _git_commit(),
        "python": platform.python_version(),
    }
    if validation is not None:
        meta["validation"] = validation
    if extra:
        meta.update(extra)

    path = meta_path_for(artifact_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def read_provenance(artifact_path: str | Path) -> Optional[dict[str, Any]]:
    """Load an artifact's sidecar, or ``None`` when it has none."""
    path = meta_path_for(artifact_path)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def describe_artifact(artifact_path: str | Path) -> str:
    """One-line human summary of an artifact's provenance, for training logs."""
    meta = read_provenance(artifact_path)
    if meta is None:
        return (f"{Path(artifact_path).name}: no provenance sidecar "
                f"(built before provenance tracking, or hand-edited)")

    gen = meta.get("generation", {})
    bits = [
        f"backend={gen.get('backend')}",
        f"model={gen.get('llm_model')}",
        f"temp={gen.get('temperature')}",
        f"prompts={gen.get('prompt_version')}",
        f"encoder={meta.get('encoder')}",
        f"hash={meta.get('content_hash')}",
    ]
    line = f"{meta.get('artifact')}: " + ", ".join(str(b) for b in bits)

    warnings = []
    if meta.get("is_dry_run"):
        warnings.append("DRY-RUN STUB TEXT")
    if not meta.get("train_filtered", True):
        warnings.append("NOT TRAIN-FILTERED (possible val/test leakage)")
    failed = (meta.get("validation") or {}).get("failed", 0)
    if failed:
        warnings.append(f"{failed} unrecoverable generations")
    if warnings:
        line += "\n    WARNING: " + "; ".join(warnings)
    return line
