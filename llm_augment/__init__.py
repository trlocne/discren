"""Offline open-source-LLM profiling pipeline.

Generates the user-profile and item-text embeddings that feed the recommender's
ProMax-style distribution-shaping loss (the only LLM signal). LLM edge
augmentation was removed.

Entry points:
  ``python -m llm_augment.build_user_profiles``
  ``python -m llm_augment.build_item_text_features``
"""

from . import prompts  # noqa: F401
