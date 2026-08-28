"""Output validation and repair for the offline LLM augmentation pipeline.

The prompts ask for one plain paragraph of a given length with no markdown and
no preamble. Nothing enforced that. An instruct model that ignores the format
(emitting ``**Profile:**`` headers, a bulleted list, an empty string, or a
truncated half-sentence) produced text that was encoded and written into the
feature matrix exactly like a good generation, so a silently degraded run was
indistinguishable from a clean one.

This module closes that gap in two stages:

* :func:`clean_generation` strips the formatting the prompt forbade — markdown
  emphasis, bullet markers, code fences, and the "Here is the profile:" style
  preamble instruct models habitually prepend.
* :func:`validate_generation` then decides whether what remains is usable,
  returning a :class:`ValidationIssue` when it is not.

:func:`generate_validated` wires both into the generation loop and *retries
only the failed prompts*, at a raised temperature so the model does not simply
reproduce its previous mistake. Whatever still fails after the retry budget is
replaced by a deterministic, clearly-marked fallback string rather than being
left as junk, and every failure is counted so the caller can report them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

_MARKDOWN_HEADER = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+", re.M)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
_CODE_FENCE = re.compile(r"^\s*```[^\n]*\n?|\n?```\s*$", re.M)
_WHITESPACE = re.compile(r"\s+")

_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is| are)\s+)?"
    r"(?:the\s+|a\s+)?"
    r"(?:shopper(?:'s)?\s+|user\s+|customer\s+|product\s+|item\s+)?"
    r"(?:preference\s+)?(?:profile|description|summary|paragraph|answer|output)"
    r"\s*[:\-\u2014]\s*",
    re.I,
)

_REFUSAL = re.compile(
    r"^\s*(?:i(?:'m| am)\s+(?:sorry|unable|not able)"
    r"|i\s+(?:cannot|can't|can not)\s"
    r"|as an ai\b"
    r"|there (?:is|are) (?:not enough|insufficient|no) (?:information|data|history))",
    re.I,
)

@dataclass
class ValidationIssue:
    """One rejected generation."""

    index: int
    reason: str
    text: str = ""

@dataclass
class ValidationStats:
    """Tally of what the validator did, for the run summary."""

    total: int = 0
    cleaned: int = 0
    retried: int = 0
    recovered: int = 0
    failed: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def summary(self) -> str:
        parts = [
            f"{self.total} generated",
            f"{self.cleaned} reformatted",
            f"{self.retried} retried",
            f"{self.recovered} recovered",
            f"{self.failed} unrecoverable",
        ]
        line = "[validate] " + ", ".join(parts)
        if self.reasons:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.reasons.items()))
            line += f"\n[validate] rejection reasons: {detail}"
        return line

def clean_generation(text: str) -> str:
    """Strip forbidden formatting and collapse the result to one paragraph."""
    if not text:
        return ""
    text = _CODE_FENCE.sub("", text)
    text = _MARKDOWN_HEADER.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()

    text = _PREAMBLE.sub("", text, count=1).strip()
    return text

def validate_generation(
    text: str,
    min_words: int = 15,
    max_words: int = 160,
) -> Optional[str]:
    """Return a rejection reason, or ``None`` when the text is acceptable.

    The word bounds are deliberately wider than the 30-80 the prompts request:
    the aim is to catch text that is *unusable* (empty, a stub, a runaway
    repetition, a refusal), not to police a model that wrote 85 words.
    """
    if not text or not text.strip():
        return "empty"
    if _REFUSAL.match(text):
        return "refusal"

    words = text.split()
    if len(words) < min_words:
        return "too_short"
    if len(words) > max_words:
        return "too_long"

    lowered = [w.lower() for w in words]
    if len(set(lowered)) < max(5, len(words) // 8):
        return "repetitive"

    if len(words) > 25 and text[-1] not in ".!?\"')":
        return "truncated"

    return None

def _fallback_text(kind: str, index: int) -> str:
    """Deterministic placeholder for a generation that could not be repaired.

    Marked with an explicit token so these rows are greppable in the audit log
    and countable after the fact, instead of masquerading as real profiles.
    """
    noun = "shopper" if kind == "user" else "product"
    return f"[GENERATION_FAILED] No usable {noun} description for id {index}."

def generate_validated(
    llm,
    system_prompt: str,
    user_prompts: list[str],
    kind: str = "user",
    max_retries: int = 2,
    retry_temperature: float = 0.7,
    min_words: int = 15,
    max_words: int = 160,
    stats: Optional[ValidationStats] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[str], ValidationStats]:
    """Generate, clean, validate, and retry the failures.

    Args:
        llm: a backend exposing ``generate(system_prompt, user_prompts)``.
        system_prompt: shared system prompt.
        user_prompts: one prompt per row.
        kind: ``"user"`` or ``"item"``, used only in the fallback text.
        max_retries: extra attempts for prompts that failed validation.
        retry_temperature: temperature for retries. Resampling at the original
            temperature tends to reproduce the same malformed output, so the
            retry deliberately explores more.
        stats: optional external tally to accumulate into.
        progress: optional callback for progress lines.

    Returns:
        ``(texts, stats)`` where ``texts`` is exactly ``len(user_prompts)``
        long, every entry either validated or an explicit failure marker.
    """
    say = progress or (lambda msg: print(msg))
    stats = stats or ValidationStats()

    raw = llm.generate(system_prompt, user_prompts)
    if len(raw) != len(user_prompts):
        raise RuntimeError(
            f"backend returned {len(raw)} generations for {len(user_prompts)} prompts"
        )

    results: list[str] = []
    pending: list[ValidationIssue] = []
    for i, text in enumerate(raw):
        cleaned = clean_generation(text)
        if cleaned != (text or "").strip():
            stats.cleaned += 1
        reason = validate_generation(cleaned, min_words, max_words)
        results.append(cleaned)
        if reason:
            pending.append(ValidationIssue(index=i, reason=reason, text=cleaned))
    stats.total += len(raw)

    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        say(f"[validate] retry {attempt}/{max_retries} for {len(pending)} "
            f"rejected generation(s)")
        stats.retried += len(pending)

        restore = _raise_temperature(llm, retry_temperature)
        try:
            retry_raw = llm.generate(
                system_prompt, [user_prompts[iss.index] for iss in pending]
            )
        finally:
            restore()

        still_bad: list[ValidationIssue] = []
        for issue, text in zip(pending, retry_raw):
            cleaned = clean_generation(text)
            reason = validate_generation(cleaned, min_words, max_words)
            if reason is None:
                results[issue.index] = cleaned
                stats.recovered += 1
            else:
                still_bad.append(
                    ValidationIssue(index=issue.index, reason=reason, text=cleaned)
                )
        pending = still_bad

    for issue in pending:
        stats.note(issue.reason)
        stats.failed += 1
        results[issue.index] = _fallback_text(kind, issue.index)

    return results, stats

def _raise_temperature(llm, temperature: float) -> Callable[[], None]:
    """Temporarily raise the backend temperature; returns an undo callable.

    Backends expose sampling differently (``self.temperature`` for the
    transformers path, a ``SamplingParams`` object for vLLM), and the echo stub
    has no temperature at all, so each case is handled and anything unknown is
    left untouched.
    """
    if hasattr(llm, "sampling") and hasattr(llm.sampling, "temperature"):
        previous = llm.sampling.temperature
        llm.sampling.temperature = temperature

        def undo_vllm() -> None:
            llm.sampling.temperature = previous

        return undo_vllm

    if hasattr(llm, "temperature"):
        previous = llm.temperature
        llm.temperature = temperature

        def undo_hf() -> None:
            llm.temperature = previous

        return undo_hf

    return lambda: None
