"""Backwards-compatible re-export.

``WeightedHypergraphConv`` now lives in :mod:`model.modules.hypergcn` alongside
the other architecture blocks. This shim keeps ``from model.hypergcn import
WeightedHypergraphConv`` working for older checkpoints, notebooks, and scripts.
"""

from model.modules.hypergcn import WeightedHypergraphConv

__all__ = ["WeightedHypergraphConv"]
