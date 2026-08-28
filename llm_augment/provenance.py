from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROMPT_VERSION = "2026-08-01.a"
META_SUFFIX = ".meta.json"


def array_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def make_sidecar(
    array: np.ndarray,
    artifact_kind: str,
    dataset: str,
    backend: str,
    encoder: str,
    prompt_version: str = PROMPT_VERSION,
    train_only: bool = True,
    sample_cap: Optional[int] = None,
    validation_summary: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "array": {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256_16": array_hash(array),
        },
        "generation": {
            "backend": backend,
            "encoder": encoder,
            "prompt_version": prompt_version,
            "train_only": bool(train_only),
            "sample_cap": sample_cap,
        },
    }
    if validation_summary is not None:
        meta["validation"] = validation_summary
    if extra:
        meta["extra"] = extra
    return meta


def write_sidecar(array_path: Path | str, sidecar: dict[str, Any]) -> Path:
    p = Path(array_path)
    meta_path = p.with_name(p.name + META_SUFFIX)
    meta_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta_path


def read_sidecar(array_path: Path | str) -> Optional[dict[str, Any]]:
    p = Path(array_path)
    meta_path = p.with_name(p.name + META_SUFFIX)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def describe_artifact(array_path: Path | str) -> str:
    p = Path(array_path)
    meta = read_sidecar(p)
    if meta is None:
        return f"{p.name} (no sidecar found next to array)"
    gen = meta.get("generation", {})
    arr = meta.get("array", {})
    val = meta.get("validation", {})
    lines = [
        f"{p.name} [{meta.get('artifact_kind', '?')}]",
        f"  shape={arr.get('shape')} dtype={arr.get('dtype')} sha256={arr.get('sha256_16')}",
        f"  backend={gen.get('backend')} encoder={gen.get('encoder')} prompt_ver={gen.get('prompt_version')}",
        f"  train_only={gen.get('train_only')} sample_cap={gen.get('sample_cap')}",
    ]
    if val:
        lines.append(f"  validation: total={val.get('total')} valid={val.get('valid')} invalid={val.get('invalid')}")
    return "\n".join(lines)
