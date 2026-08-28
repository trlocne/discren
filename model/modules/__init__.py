"""DISCREN architecture modules.

The package layout mirrors the architecture figure one-to-one, so a block in
the paper maps to exactly one file here:

    Figure block                       Module
    ---------------------------------- -------------------------------------
    Degree-scaled DropEdge             graph_ops.DegreeScaledDropEdge
    Popularity gate  (alpha)           graph_ops.PopularityGate
    Collaborative backbone (B1-B3)     collaborative.CollaborativeBackbone
    Reciprocal cross-modal attention   rca.ReciprocalCrossModalAttention
    Weighted hypergraph convolution    hypergcn.WeightedHypergraphConv
    Multimodal semantic encoder (M1-M4) multimodal.MultimodalSemanticEncoder
    Controlled LLM injection (L1-L4)   llm_injection.ControlledLLMInjection
    Training objective (all terms)     losses.*

Every module exposes a ``debug`` dict that is refreshed on each forward pass.
Reading ``model.debug_state()`` after a step therefore gives a single flat view
of every intermediate tensor statistic without adding print statements.
"""

from model.modules.collaborative import CollaborativeBackbone
from model.modules.graph_ops import (
    DegreeScaledDropEdge,
    PopularityGate,
    propagate,
)
from model.modules.hypergcn import WeightedHypergraphConv
from model.modules.llm_injection import ControlledLLMInjection
from model.modules.multimodal import MultimodalSemanticEncoder
from model.modules.rca import ReciprocalCrossModalAttention

__all__ = [
    "CollaborativeBackbone",
    "ControlledLLMInjection",
    "DegreeScaledDropEdge",
    "MultimodalSemanticEncoder",
    "PopularityGate",
    "ReciprocalCrossModalAttention",
    "WeightedHypergraphConv",
    "propagate",
]
