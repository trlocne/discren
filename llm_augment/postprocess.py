from __future__ import annotations
from typing import Optional
import numpy as np

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)

def all_but_the_top(matrix: np.ndarray, n_components: int=1) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if n_components <= 0:
        return centered.astype(np.float32)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components]
    return (centered - centered @ basis.T @ basis).astype(np.float32)

def pca_whiten(matrix: np.ndarray, alpha: float=1.0) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if alpha <= 0:
        return centered.astype(np.float32)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scaled = s ** (1.0 - float(alpha))
    return (u * scaled @ vt).astype(np.float32)

class FeaturePostprocessor:

    def __init__(self, method: str='whiten', n_components: int=1, alpha: float=1.0, normalize: bool=True):
        valid = {'none', 'center', 'abtt', 'whiten'}
        if method not in valid:
            raise ValueError(f'method must be one of {sorted(valid)}, got {method!r}')
        self.method = method
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.normalize = bool(normalize)
        self.mean_: Optional[np.ndarray] = None
        self.transform_: Optional[np.ndarray] = None

    def fit(self, matrix: np.ndarray) -> 'FeaturePostprocessor':
        matrix = np.asarray(matrix, dtype=np.float32)
        nonzero = np.linalg.norm(matrix, axis=1) > 1e-08
        fit_rows = matrix[nonzero] if nonzero.any() else matrix
        self.mean_ = fit_rows.mean(axis=0, keepdims=True).astype(np.float32)
        if self.method in ('none', 'center'):
            self.transform_ = None
            return self
        centered = fit_rows - self.mean_
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        if self.method == 'abtt':
            basis = vt[:self.n_components]
            self.transform_ = (np.eye(matrix.shape[1], dtype=np.float32) - basis.T @ basis).astype(np.float32)
        else:
            scale = np.zeros_like(s)
            nz = s > 1e-08
            scale[nz] = s[nz] ** (-self.alpha)
            self.transform_ = (vt.T * scale @ vt).astype(np.float32)
        return self

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError('FeaturePostprocessor.apply called before fit')
        matrix = np.asarray(matrix, dtype=np.float32)
        if self.method == 'none':
            out = matrix.copy()
        else:
            zero = np.linalg.norm(matrix, axis=1) < 1e-08
            out = matrix - self.mean_
            if self.transform_ is not None:
                out = out @ self.transform_
            out[zero] = 0.0
        return l2_normalize(out) if self.normalize else out.astype(np.float32)

    def fit_apply(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).apply(matrix)

    def describe(self) -> str:
        if self.method in ('none', 'center'):
            return f'postprocess={self.method}'
        if self.method == 'abtt':
            return f'postprocess=abtt(k={self.n_components})'
        return f'postprocess=whiten(alpha={self.alpha})'

def build_postprocessor(method: str='whiten', n_components: int=1, alpha: float=1.0, normalize: bool=True) -> FeaturePostprocessor:
    return FeaturePostprocessor(method=method, n_components=n_components, alpha=alpha, normalize=normalize)
