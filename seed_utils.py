"""Seed-aware checkpoint paths, shared by ``main_modal.py`` and ``eval_modal.py``.

A single training run cannot distinguish a real effect from run-to-run noise.
The measured full-vs-no-LLM gap on Clothing is about +1.2% Recall@20, which is
well inside the range a seed change can produce, so the comparison has to be
repeated over several seeds.

To make that possible without runs overwriting each other, every seeded run
gets its own checkpoint directory::

    /checkpoints/full            # seed taken from the YAML (legacy layout)
    /checkpoints/full/seed7      # --seed 7
    /checkpoints/full/seed13     # --seed 13

Training and evaluation must agree on that path, hence this single helper
rather than two copies of the same string manipulation.
"""

from __future__ import annotations

import os


def resolve_save_dir(base_dir: str, seed_override: int) -> str:
    """Return the checkpoint directory for one seeded run.

    Args:
        base_dir: ``training.save_dir`` from the YAML config.
        seed_override: the CLI ``--seed`` value. Negative means "not
            overridden", in which case the legacy un-suffixed path is used so
            existing checkpoints keep resolving.

    Returns:
        ``base_dir`` when no override was given, else ``base_dir/seed<N>``.
    """
    if seed_override is None or int(seed_override) < 0:
        return base_dir
    return os.path.join(base_dir, f"seed{int(seed_override)}")
