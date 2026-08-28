"""Post-processing that makes LLM sentence embeddings usable for retrieval.

Sentence encoders place all their outputs in a narrow cone: on this corpus two
*random* user profiles have cosine 0.64 and two random LLM item descriptions
0.49, versus 0.42 for the raw text features. A large shared component like that
dominates every dot product, so nearest-neighbour search is driven mostly by
what all the texts have in common rather than by what distinguishes them.

The injection block cannot undo this. ``ControlledLLMInjection`` normalizes the
*projection* of the feature, and a linear map followed by L2 normalization can
rotate and rescale the cone but cannot remove its axis, so the anisotropy
propagates into the model.

Measured on Clothing with co-purchase pairs as ground truth
(``tools/probe_feature_space.py``), on ``text_llm_feat.npy``:

======================  ======  ========
transform               d       hit@10
======================  ======  ========
raw                     0.550   33.67%
mean-centered           0.483   33.73%
all-but-the-top k=1     0.423   33.87%
all-but-the-top k=3     0.322   31.13%
pca_whiten alpha=0.25   0.468   36.27%
pca_whiten alpha=0.5    0.424   39.13%
pca_whiten alpha=1.0    0.206   39.73%
======================  ======  ========

``d`` falls monotonically while ``hit@10`` rises by up to 6 points. That is not
a contradiction: ``d`` is a separation of *raw* cosines, so removing the shared
component also removes the offset that inflated it. ``hit@10`` asks whether the
top-k neighbourhood -- precisely what the kNN and cluster-incidence hypergraphs
are built from -- contains genuine co-purchase partners. The retrieval metric
reflects what the recommender actually consumes, so judging by ``d`` alone
would have pointed the wrong way.

Note also that ``all-but-the-top`` is *not* a cheap substitute here: at k=1 it
barely moves ``hit@10``, and by k=3 it is removing useful directions. The gain
comes from flattening the whole spectrum, not from deleting a few axes. Most of
it is already available at ``alpha=0.5`` (39.13% vs 39.73%), which is the safer
default on a noisier corpus since full whitening also amplifies the
lowest-variance directions.

All transforms are fitted on the rows they are given. Callers that hold rows out
must fit on the training rows and apply the stored statistics to the rest;
:class:`FeaturePostprocessor` exists for that.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization that leaves all-zero rows at zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)

def all_but_the_top(matrix: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Mean-center, then project out the top ``n_components`` directions.

    Mu et al., "All-but-the-Top: Simple and Effective Postprocessing for Word
    Representations" (ICLR 2018). Cheaper than whitening and it preserves the
    relative scale of the surviving directions, but on this corpus it recovers
    less of the available gain than partial whitening does.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if n_components <= 0:
        return centered.astype(np.float32)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components]
    return (centered - (centered @ basis.T) @ basis).astype(np.float32)

def pca_whiten(matrix: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Mean-center and flatten the singular-value spectrum by exponent ``alpha``.

    Component ``j`` is rescaled by ``sigma_j ** (1 - alpha)``:

    * ``alpha = 0`` leaves the spectrum untouched (mean-centering only),
    * ``alpha = 1`` is full whitening, giving every direction equal variance,
    * intermediate values interpolate.

    Full whitening scores best on ``hit@10`` here, but it also amplifies the
    lowest-variance directions, which on a smaller or noisier corpus is mostly
    amplifying noise. ``alpha`` is exposed so that trade-off stays a decision
    rather than a hidden constant.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if alpha <= 0:
        return centered.astype(np.float32)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scaled = s ** (1.0 - float(alpha))
    return ((u * scaled) @ vt).astype(np.float32)

class FeaturePostprocessor:
    """Fit a post-processing transform once, then apply it to any rows.

    Keeping the fitted statistics separate from their application is what makes
    a leakage-free setup possible: fit on training rows, transform everything.
    Fitting on all rows would let held-out rows influence the mean and the
    principal directions the training features are expressed in.

    Args:
        method: ``"none"``, ``"center"``, ``"abtt"`` or ``"whiten"``.
        n_components: number of directions removed by ``"abtt"``.
        alpha: whitening strength for ``"whiten"``.
        normalize: L2-normalize rows after transforming.
    """

    def __init__(
        self,
        method: str = "whiten",
        n_components: int = 1,
        alpha: float = 1.0,
        normalize: bool = True,
    ):
        valid = {"none", "center", "abtt", "whiten"}
        if method not in valid:
            raise ValueError(f"method must be one of {sorted(valid)}, got {method!r}")
        self.method = method
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.normalize = bool(normalize)
        self.mean_: Optional[np.ndarray] = None
        self.transform_: Optional[np.ndarray] = None

    def fit(self, matrix: np.ndarray) -> "FeaturePostprocessor":
        """Learn the mean and the linear map from ``matrix``.

        Rows that are exactly zero are excluded from the fit: they are padding
        for entities with no generated text, not observations, and including
        them would drag the mean toward the origin in proportion to how many
        happen to be present.
        """
        matrix = np.asarray(matrix, dtype=np.float32)
        nonzero = np.linalg.norm(matrix, axis=1) > 1e-8
        fit_rows = matrix[nonzero] if nonzero.any() else matrix

        self.mean_ = fit_rows.mean(axis=0, keepdims=True).astype(np.float32)
        if self.method in ("none", "center"):
            self.transform_ = None
            return self

        centered = fit_rows - self.mean_
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)

        if self.method == "abtt":
            basis = vt[: self.n_components]

            self.transform_ = (
                np.eye(matrix.shape[1], dtype=np.float32) - basis.T @ basis
            ).astype(np.float32)
        else:
            scale = np.zeros_like(s)
            nz = s > 1e-8
            scale[nz] = s[nz] ** (-self.alpha)
            self.transform_ = ((vt.T * scale) @ vt).astype(np.float32)
        return self

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Transform ``matrix`` using the statistics learned by :meth:`fit`."""
        if self.mean_ is None:
            raise RuntimeError("FeaturePostprocessor.apply called before fit")
        matrix = np.asarray(matrix, dtype=np.float32)
        if self.method == "none":
            out = matrix.copy()
        else:

            zero = np.linalg.norm(matrix, axis=1) < 1e-8
            out = matrix - self.mean_
            if self.transform_ is not None:
                out = out @ self.transform_
            out[zero] = 0.0
        return l2_normalize(out) if self.normalize else out.astype(np.float32)

    def fit_apply(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).apply(matrix)

    def describe(self) -> str:
        if self.method in ("none", "center"):
            return f"postprocess={self.method}"
        if self.method == "abtt":
            return f"postprocess=abtt(k={self.n_components})"
        return f"postprocess=whiten(alpha={self.alpha})"

def build_postprocessor(
    method: str = "whiten",
    n_components: int = 1,
    alpha: float = 1.0,
    normalize: bool = True,
) -> FeaturePostprocessor:
    """Factory mirroring the CLI/config surface."""
    return FeaturePostprocessor(
        method=method, n_components=n_components, alpha=alpha, normalize=normalize
    )
