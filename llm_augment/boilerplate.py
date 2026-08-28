from __future__ import annotations
import collections
import re
_SUBJECT = '(?:this|the)\\s+(?:shopper|user|customer|buyer|reviewer)'
_ADVERB = '(?:consistently|frequently|repeatedly|often|regularly|primarily|mainly|mostly|typically|generally|habitually|predominantly|largely|clearly|strongly|actively)'
_VERB = '(?:purchases|purchased|buys|bought|selects|chooses|seeks|prefers|favors|favours|shops|engages\\s+with|gravitates\\s+toward|tends\\s+to\\s+(?:buy|purchase|choose|select|prefer)|shows?\\s+a\\s+preference\\s+for|demonstrates?\\s+a\\s+preference\\s+for|is\\s+drawn\\s+to|focuses\\s+on|prioritizes|prioritises|values)'
_OPENING = re.compile(f'^\\s*{_SUBJECT}\\s+(?:{_ADVERB}\\s+)?{_VERB}\\s+', re.I)
_ITEM_SUBJECT = '(?:this|the)\\s+(?:product|item|garment|piece|shoe|bag|watch)'
_ITEM_OPENING = re.compile(f'^\\s*{_ITEM_SUBJECT}\\s+(?:is\\s+(?:a|an)\\s+|offers\\s+|features\\s+|provides\\s+|comes\\s+(?:in|with)\\s+)', re.I)

def strip_boilerplate_opening(text: str, kind: str='user') -> str:
    if not text:
        return text
    pattern = _ITEM_OPENING if kind == 'item' else _OPENING
    stripped = pattern.sub('', text, count=1)
    if stripped is text or not stripped.strip():
        return text
    stripped = stripped.lstrip()
    return stripped[0].upper() + stripped[1:] if stripped else text

def strip_all(texts: list[str], kind: str='user') -> tuple[list[str], int]:
    out: list[str] = []
    changed = 0
    for text in texts:
        new = strip_boilerplate_opening(text, kind)
        changed += new != text
        out.append(new)
    return (out, changed)

def opening_diversity(texts: list[str], n_words: int=4) -> dict[str, float]:
    if not texts:
        return {'distinct_ratio': 0.0, 'top1_share': 0.0, 'top5_share': 0.0}
    counts = collections.Counter((' '.join(t.lower().split()[:n_words]) for t in texts))
    total = len(texts)
    ranked = counts.most_common()
    return {'distinct_ratio': len(counts) / total, 'top1_share': ranked[0][1] / total, 'top5_share': sum((c for _, c in ranked[:5])) / total}

def format_diversity(stats: dict[str, float]) -> str:
    return f"distinct_openings={100 * stats['distinct_ratio']:.1f}%  top1={100 * stats['top1_share']:.1f}%  top5={100 * stats['top5_share']:.1f}%"
