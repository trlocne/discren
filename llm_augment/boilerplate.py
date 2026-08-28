"""Strip formulaic openings before generated text is encoded.

The prompts ask the model to vary its phrasing. It does not. Across the 39387
generated user profiles:

    44.50%  "this shopper consistently purchases"
    21.60%  "this shopper consistently buys"
     4.04%  "this shopper frequently purchases"
     3.76%  "this shopper consistently selects"

Two openings account for 66% of the corpus, and 15 tokens appear in more than
half of all profiles, making up 30.2% of the tokens in an average one. A
sentence encoder has no way to know that this shared preamble is scaffolding
rather than content, so it spends a large part of the embedding budget encoding
text that is identical for tens of thousands of users -- which is a direct
contributor to the 0.64 mean pairwise cosine those embeddings exhibit.

Removing the scaffold at encode time is the cheap half of the fix: it needs no
regeneration and touches only what the encoder sees. The audit log keeps the
original text, so nothing is lost for inspection.

This is deliberately conservative. It removes only leading discourse framing
("This shopper consistently purchases ...", "The user tends to buy ...") and
never touches the nouns that follow, because those carry the signal. Text that
does not begin with a recognised pattern is returned unchanged.
"""

from __future__ import annotations

import collections
import re

_SUBJECT = r"(?:this|the)\s+(?:shopper|user|customer|buyer|reviewer)"

_ADVERB = (r"(?:consistently|frequently|repeatedly|often|regularly|primarily|"
           r"mainly|mostly|typically|generally|habitually|predominantly|"
           r"largely|clearly|strongly|actively)")

_VERB = (r"(?:purchases|purchased|buys|bought|selects|chooses|seeks|prefers|"
         r"favors|favours|shops|engages\s+with|gravitates\s+toward|"
         r"tends\s+to\s+(?:buy|purchase|choose|select|prefer)|"
         r"shows?\s+a\s+preference\s+for|demonstrates?\s+a\s+preference\s+for|"
         r"is\s+drawn\s+to|focuses\s+on|prioritizes|prioritises|values)")

_OPENING = re.compile(
    rf"^\s*{_SUBJECT}\s+(?:{_ADVERB}\s+)?{_VERB}\s+", re.I
)

_ITEM_SUBJECT = r"(?:this|the)\s+(?:product|item|garment|piece|shoe|bag|watch)"
_ITEM_OPENING = re.compile(
    rf"^\s*{_ITEM_SUBJECT}\s+(?:is\s+(?:a|an)\s+|offers\s+|features\s+|"
    rf"provides\s+|comes\s+(?:in|with)\s+)",
    re.I,
)

def strip_boilerplate_opening(text: str, kind: str = "user") -> str:
    """Remove a formulaic opening clause, if present.

    Args:
        text: one generated profile or description.
        kind: ``"user"`` or ``"item"``, selecting the pattern set.

    Returns:
        The text with its opening scaffold removed and the first character
        re-capitalized. Unrecognised text is returned unchanged.
    """
    if not text:
        return text
    pattern = _ITEM_OPENING if kind == "item" else _OPENING
    stripped = pattern.sub("", text, count=1)
    if stripped is text or not stripped.strip():

        return text
    stripped = stripped.lstrip()
    return stripped[0].upper() + stripped[1:] if stripped else text

def strip_all(texts: list[str], kind: str = "user") -> tuple[list[str], int]:
    """Apply :func:`strip_boilerplate_opening` to a corpus.

    Returns:
        ``(stripped_texts, n_changed)``.
    """
    out: list[str] = []
    changed = 0
    for text in texts:
        new = strip_boilerplate_opening(text, kind)
        changed += new != text
        out.append(new)
    return out, changed

def opening_diversity(texts: list[str], n_words: int = 4) -> dict[str, float]:
    """Summarize how repetitive the corpus openings are.

    Returned keys:

    ``distinct_ratio``
        distinct openings divided by number of texts; 1.0 means every text
        starts differently.
    ``top1_share`` / ``top5_share``
        share of the corpus covered by the single most common opening, and by
        the five most common.
    """
    if not texts:
        return {"distinct_ratio": 0.0, "top1_share": 0.0, "top5_share": 0.0}
    counts = collections.Counter(
        " ".join(t.lower().split()[:n_words]) for t in texts
    )
    total = len(texts)
    ranked = counts.most_common()
    return {
        "distinct_ratio": len(counts) / total,
        "top1_share": ranked[0][1] / total,
        "top5_share": sum(c for _, c in ranked[:5]) / total,
    }

def format_diversity(stats: dict[str, float]) -> str:
    return (f"distinct_openings={100 * stats['distinct_ratio']:.1f}%  "
            f"top1={100 * stats['top1_share']:.1f}%  "
            f"top5={100 * stats['top5_share']:.1f}%")
