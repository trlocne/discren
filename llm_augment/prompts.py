"""
Default prompt templates for the offline LLM data-augmentation pipeline.

These prompts implement the *LLM Semantic Profiling* module (A2 in the paper):
an open-source LLM reads a user's interaction history (item titles / categories
/ their own reviews) and produces a concise natural-language *preference
profile*. The profile is later encoded by a sentence encoder into
``user_profile_feat.npy`` — one row per user, in the exact order of
``user_list.txt`` (mirroring how ``text_feat.npy`` mirrors ``item_list.txt``).

The prompts are intentionally domain-agnostic but tuned for the Amazon
*Clothing, Shoes & Jewelry* corpus used in this project. Override any of them
from your own script if you switch domains.
"""

from __future__ import annotations

from textwrap import dedent


# ─────────────────────────────────────────────────────────────────────────────
# A2 — User semantic-profile generation
# ─────────────────────────────────────────────────────────────────────────────

# Domain descriptions, injected into both system prompts. Telling the model it is
# looking at a "fashion marketplace" while showing it camping stoves and fishing
# reels invites exactly the ungrounded inference the prompts otherwise work to
# suppress, so the domain is per-dataset rather than hardcoded.
DOMAINS = {
    "clothing": "a fashion marketplace (clothing, shoes and jewelry)",
    "sports": "a sports and outdoors marketplace (fitness equipment, outdoor "
               "gear, apparel and accessories)",
    "generic": "an online retail marketplace",
}
DEFAULT_DOMAIN = "clothing"


def resolve_domain(name: str | None) -> str:
    """Map a dataset name to its domain phrase, defaulting to a neutral one."""
    if not name:
        return DOMAINS[DEFAULT_DOMAIN]
    return DOMAINS.get(str(name).strip().lower(), DOMAINS["generic"])


USER_PROFILE_SYSTEM_PROMPT = dedent(
    """\
    You are an expert e-commerce preference analyst. Given the purchase and
    review history of a single anonymous shopper on {domain}, you summarise WHO
    this shopper is as a consumer — in a way that would let someone tell this
    shopper apart from a DIFFERENT shopper with a similar-looking history.

    Grounding rules (avoid generic filler):
    - Every claim must be traceable to the history: if a brand, category,
      colour, material or price tier appears more than once, name it
      explicitly (e.g. "repeatedly buys Nike running shoes", not just
      "prefers athletic wear"). Prefer concrete nouns over abstract
      adjectives whenever the history supports them.
    - Do NOT default to filler phrases like "mid-range priced", "casual and
      versatile", "values comfort and quality" unless the history gives no
      more specific signal at all — these phrases are true of most shoppers
      and add no distinguishing information. If you catch yourself about to
      write one, look again for a more specific detail (a repeated brand, an
      unusual category mix, a distinctive complaint or compliment) instead.
    - Lead with whatever is MOST DISTINCTIVE about this particular shopper
      (an unusual combination of categories, a strong like/dislike, a clear
      recipient other than themselves, a narrow niche), not with a generic
      opening sentence. Different shoppers should read differently even if
      both happen to shop mid-range casual wear.
    - Infer sentiment from the reviews (what they like vs. dislike) and state
      it concretely (what specifically worked or failed), not just as
      "positive/negative sentiment".
    - Do NOT list individual product ids, do NOT invent facts not supported by
      the history, do NOT add greetings, headers, markdown or bullet points.

    Format:
    - ONE dense paragraph, 40-80 words, third person ("This shopper ...").
    - Vary sentence structure and word choice between profiles — do not reuse
      the same opening template ("This shopper prefers...") for every writeup;
      start instead with whatever detail is most distinctive for THIS shopper.
    - Output ONLY the profile paragraph, nothing else.
    """
)

USER_PROFILE_USER_PROMPT = dedent(
    """\
    Below is the interaction history of one shopper. Each line is an item they
    interacted with, optionally followed by the rating they gave and a short
    excerpt of their review.

    ---
    {history_block}
    ---

    First, silently identify: (a) any brand, category, colour or material that
    repeats across multiple lines, (b) the most distinctive or unusual thing
    about this shopper compared to a typical shopper, and (c) any concrete
    likes/dislikes stated in the reviews. Then write the shopper's preference
    profile now, grounded in those specifics rather than generic descriptors.
    """
)


# ─────────────────────────────────────────────────────────────────────────────
# Item-side text enrichment (only needed if you want to (re)build text_feat.npy)
# ─────────────────────────────────────────────────────────────────────────────
# Goal: produce a short, *retrieval-friendly* item description to encode into
# the item text-modality embedding used by the recommender (mirrors the role
# of a product title in content-based / hybrid recsys pipelines).
#
# Amazon 5-core review dumps carry no title, but the companion
# ``metadata.json.gz`` does (title, brand, categories, price). When available,
# metadata is the primary, authoritative source; reviews are used ONLY to fill
# gaps (style/fit/quality signals a bare title doesn't convey) and are
# strictly secondary — never allowed to override or contradict metadata.
# ``{metadata_block}`` is empty/omitted for items with no metadata match, in
# which case the model must rely on reviews alone (existing behaviour).

ITEM_PROFILE_SYSTEM_PROMPT = dedent(
    """\
    You are a catalog content specialist preparing product text for the
    RECOMMENDER SYSTEM of {domain}. Your output will be embedded into
    a vector and used for item-item similarity and retrieval, so it must be
    dense with the concrete, discriminative attributes a recommendation model
    can key on: product type, category/subcategory, intended user (men /
    women / kids / unisex), style, color, material, fit or sizing,
    occasion/use-case, and price tier.

    You are given structured catalog metadata (title, brand, category path,
    price) and, optionally, a few customer reviews for the SAME product.

    Rules:
    - ONE paragraph, 30-70 words, third person, no marketing hype, no
      first/second person, no emojis, no markdown.
    - Treat metadata (title, brand, category, price) as ground truth. Use
      reviews only to add attributes not already covered by metadata (e.g.
      fit, comfort, durability, sizing accuracy) — never contradict metadata.
    - If reviews conflict with each other, prefer the majority sentiment.
    - Mention brand and category naturally if present; mention price tier
      qualitatively (budget / mid-range / premium) instead of the raw number.
    - Do NOT invent attributes not supported by the given metadata or reviews.
    - Output ONLY the description paragraph, nothing else.
    """
)

ITEM_PROFILE_USER_PROMPT = dedent(
    """\
    Catalog metadata for this product:
    ---
    {metadata_block}
    ---

    Customer reviews for this product (may be empty):
    ---
    {review_block}
    ---

    Write the recommender-system product description now.
    """
)


# NOTE: The LLM edge-augmentation prompts (LLMRec strategy A) were removed, and
# so was the ProMax/SDR distribution-shaping loss an earlier revision of this
# comment referred to. The recommender consumes the user-profile and item-text
# embeddings only through the controlled injection blocks in
# model/modules/llm_injection.py (L1-L5) and their masked-reconstruction loss.
#
# Prompt changes alter every generated artifact. Bump PROMPT_VERSION in
# llm_augment/provenance.py whenever the text below changes materially, so two
# artifacts built from different prompts are distinguishable after the fact.


def build_item_metadata_block(meta: dict | None) -> str:
    """Render one item's catalog metadata into ``{metadata_block}``.

    ``meta`` matches the dict shape returned by
    ``data_utils.load_item_metadata``: ``title``, ``brand``,
    ``categories`` (list[list[str]]), ``price`` (float|None).
    """
    if not meta:
        return "(no catalog metadata available for this product)"

    lines: list[str] = []
    title = meta.get("title")
    if title:
        lines.append(f"Title: {title}")
    brand = meta.get("brand")
    if brand:
        lines.append(f"Brand: {brand}")
    categories = meta.get("categories") or []
    if categories:
        # categories is a list of paths, e.g. [["Clothing", "Women", "Dresses"]]
        flat_paths = [" > ".join(path) for path in categories if path]
        if flat_paths:
            lines.append(f"Category: {'; '.join(flat_paths)}")
    price = meta.get("price")
    if price is not None:
        lines.append(f"Price: ${price:.2f}")

    return "\n".join(lines) if lines else "(no catalog metadata available for this product)"


def build_user_history_block(records: list[dict], max_items: int = 20,
                             max_review_chars: int = 160) -> str:
    """Render a user's interaction records into the ``{history_block}`` string.

    Each record is a dict with keys: ``item_desc`` (str), ``rating`` (float|None),
    ``review`` (str|None). Only the first ``max_items`` are used (most recent
    first is recommended by the caller).
    """
    lines: list[str] = []
    for rec in records[:max_items]:
        desc = (rec.get("item_desc") or f"item {rec.get('item_id', '?')}").strip()
        parts = [f"- {desc}"]
        rating = rec.get("rating")
        if rating is not None:
            parts.append(f"(rated {rating:g}/5)")
        review = (rec.get("review") or "").strip().replace("\n", " ")
        if review:
            if len(review) > max_review_chars:
                review = review[:max_review_chars].rstrip() + "…"
            parts.append(f'— "{review}"')
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "- (no recorded interactions)"
