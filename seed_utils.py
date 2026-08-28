from __future__ import annotations
import os

def resolve_save_dir(base_dir: str, seed_override: int) -> str:
    if seed_override is None or int(seed_override) < 0:
        return base_dir
    return os.path.join(base_dir, f'seed{int(seed_override)}')
