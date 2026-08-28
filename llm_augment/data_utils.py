"""
Data-loading utilities for the offline LLM augmentation pipeline.

Everything here is keyed on the *integer ids* produced by ``item_list.txt`` /
``user_list.txt`` so that any embedding we generate is row-aligned with the
existing ``text_feat.npy`` / ``image_feat.npy`` and with the train/val/test
splits.

Files consumed (all under ``data/Clothing/``)::

    item_list.txt   asin<TAB>item_id          (23033 rows)
    user_list.txt   reviewerID<TAB>user_id    (39387 rows)

Raw reviews (under ``data/Clothing_raw/``)::

    reviews_Clothing_Shoes_and_Jewelry_5.json.gz
        one JSON object per line with keys:
        reviewerID, asin, reviewText, overall, summary, unixReviewTime
"""

from __future__ import annotations

import ast
import gzip
import json
import re
from pathlib import Path

def load_id_map(path: str | Path) -> dict[str, int]:
    """Load a ``<key>\\t<int_id>`` mapping file into ``{key: id}``."""
    mapping: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key, idx = line.split("\t")
            mapping[key] = int(idx)
    return mapping

def _open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def load_train_pairs(train_json_path: str | Path) -> set[tuple[int, int]]:
    """Load ``{user_id: [item_id, ...]}`` train split into a set of (uid, iid).

    The train JSON already uses the *integer* ids that match
    ``item_list.txt`` / ``user_list.txt``, so no re-mapping is needed. This is
    used to restrict LLM profiling to TRAIN interactions only and avoid
    leaking val/test signal into the generated features.
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    pairs: set[tuple[int, int]] = set()
    for uid_str, items in train.items():
        uid = int(uid_str)
        for iid in items:
            pairs.add((uid, int(iid)))
    return pairs

def load_user_histories(
    raw_reviews_path: str | Path,
    user_map: dict[str, int],
    item_map: dict[str, int],
    max_items_per_user: int = 20,
    allowed_pairs: set[tuple[int, int]] | None = None,
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Group raw reviews by user id and by item id.

    Parameters
    ----------
    allowed_pairs : set[(user_id, item_id)] | None
        If provided (e.g. from ``load_train_pairs``), only interactions whose
        ``(uid, iid)`` is in this set are kept. This restricts profiling to the
        TRAIN split and prevents val/test leakage. ``None`` keeps every review.

    Returns
    -------
    user_histories : list[list[dict]]
        Indexed by user id. Each element is a list of interaction records
        ``{item_id, asin, rating, review}`` sorted most-recent-first and
        truncated to ``max_items_per_user``.
    item_reviews : list[list[dict]]
        Indexed by item id. Each element is a list of ``{review, rating}``
        written about that item (used for optional item-side enrichment).
    """
    n_users = max(user_map.values()) + 1
    n_items = max(item_map.values()) + 1

    per_user: list[list[tuple[int, dict]]] = [[] for _ in range(n_users)]
    per_item: list[list[dict]] = [[] for _ in range(n_items)]

    path = Path(raw_reviews_path)
    n_lines = 0
    n_skipped = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            reviewer = obj.get("reviewerID")
            asin = obj.get("asin")
            uid = user_map.get(reviewer)
            iid = item_map.get(asin)
            if uid is None or iid is None:
                n_skipped += 1
                continue

            if allowed_pairs is not None and (uid, iid) not in allowed_pairs:
                n_skipped += 1
                continue

            rating = obj.get("overall")
            summary = (obj.get("summary") or "").strip()
            body = (obj.get("reviewText") or "").strip()
            review = (summary + ". " + body).strip(". ").strip() if summary else body
            ts = int(obj.get("unixReviewTime") or 0)

            rec = {
                "item_id": iid,
                "asin": asin,
                "rating": float(rating) if rating is not None else None,
                "review": review,
            }
            per_user[uid].append((ts, rec))
            per_item[iid].append({"review": review, "rating": rec["rating"]})

    user_histories: list[list[dict]] = []
    for recs in per_user:
        recs.sort(key=lambda x: x[0], reverse=True)
        user_histories.append([r for _, r in recs[:max_items_per_user]])

    print(
        f"[data_utils] parsed {n_lines} reviews "
        f"({n_skipped} skipped), "
        f"{n_users} users, {n_items} items"
    )
    return user_histories, per_item

def load_item_descriptions(item_desc_path: str | Path | None,
                           n_items: int) -> list[str | None]:
    """Load optional per-item textual descriptions (item_id -> text).

    The file, if provided, is ``<item_id>\\t<description>`` per line. Missing
    items get ``None`` so callers can fall back to ``"item <id>"``.
    """
    descs: list[str | None] = [None] * n_items
    if item_desc_path is None:
        return descs
    with open(item_desc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            idx, text = line.split("\t", 1)
            i = int(idx)
            if 0 <= i < n_items:
                descs[i] = text
    return descs

def _parse_metadata_line(line: str) -> dict | None:
    """Parse one line of the Amazon ``metadata.json.gz`` dump.

    These files are NOT strict JSON — each line is a Python dict literal
    (single quotes, no quoted keys guarantee), so we try ``json.loads`` first
    and fall back to ``ast.literal_eval``.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None

def _parse_price(raw) -> float | None:
    """Parse a price field that may be a number or a formatted string.

    The 2014 Amazon dumps (Clothing) store ``price`` as a bare float. The 2018
    dumps (Sports) store it as a display string such as ``"$11.80"``, and
    sometimes as a range like ``"$9.99 - $19.99"``. A plain ``float()`` returns
    ``None`` for every one of those, which silently drops the price from the
    prompt for the majority of items, so the currency symbol and any range are
    stripped and the first parseable number is used.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None

    match = re.search(r"\d+(?:[.,]\d+)?", text.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

def _parse_categories(obj: dict) -> list[list[str]]:
    """Normalize the two Amazon category schemas to ``list[list[str]]``.

    The 2014 dumps expose ``categories`` as a list of paths, e.g.
    ``[["Clothing", "Women", "Dresses"]]``. The 2018 dumps replaced it with a
    single flat ``category`` list, e.g.
    ``["Sports & Outdoors", "Sports & Fitness", "Dance"]``. Reading only
    ``categories`` therefore yields an empty category for every item in a 2018
    dump; on Sports that is 93% of sampled records. Both shapes are accepted
    and returned in the nested form the prompt builders already expect.
    """
    nested = obj.get("categories")
    if nested:

        if isinstance(nested[0], (list, tuple)):
            return [list(path) for path in nested if path]
        return [[str(x) for x in nested]]
    flat = obj.get("category")
    if flat:
        if isinstance(flat, (list, tuple)):
            return [[str(x) for x in flat]]
        return [[str(flat)]]
    return []

def load_item_metadata(
    metadata_path: str | Path | None,
    item_map: dict[str, int],
    n_items: int,
) -> list[dict | None]:
    """Load per-item catalog metadata keyed by asin -> item_id.

    Returns a list indexed by item id, each entry either ``None`` (no
    metadata found) or a dict with keys ``title``, ``brand``, ``categories``
    (list[list[str]]), ``price`` (float|None). Only the fields relevant to
    text enrichment are kept — ``related``/``salesRank``/``imUrl`` are dropped.

    Handles both the 2014 and 2018 Amazon metadata schemas; see
    :func:`_parse_categories` and :func:`_parse_price` for what differs.
    """
    metas: list[dict | None] = [None] * n_items
    if metadata_path is None:
        return metas
    path = Path(metadata_path)
    if not path.exists():
        print(f"[data_utils] metadata file not found: {path} — skipping metadata enrichment")
        return metas

    n_matched = 0
    n_title = 0
    n_cat = 0
    n_price = 0
    with _open_maybe_gzip(path) as f:
        for line in f:
            obj = _parse_metadata_line(line)
            if obj is None:
                continue
            asin = obj.get("asin")
            iid = item_map.get(asin)
            if iid is None or not (0 <= iid < n_items):
                continue
            title = (obj.get("title") or "").strip() or None
            categories = _parse_categories(obj)
            price = _parse_price(obj.get("price"))
            metas[iid] = {
                "title": title,
                "brand": (obj.get("brand") or "").strip() or None,
                "categories": categories,
                "price": price,
            }
            n_matched += 1
            n_title += title is not None
            n_cat += bool(categories)
            n_price += price is not None

    print(f"[data_utils] matched metadata for {n_matched}/{n_items} items "
          f"(title={n_title}, category={n_cat}, price={n_price})")
    if n_matched and n_cat == 0:
        print("[data_utils] WARNING: no categories parsed from any record. The "
              "metadata schema may be unrecognized, which would strip the "
              "category path from every prompt.")
    return metas

def build_item_desc_from_metadata(meta: dict | None) -> str | None:
    """Compact one-line item description from catalog metadata.

    Used as a fallback for user-history prompts (``build_user_history_block``)
    when no LLM-generated item description (``item_profiles.txt``) is
    available yet. Returns ``None`` if there is nothing usable (caller then
    falls back to ``"item <id>"``).
    """
    if not meta:
        return None
    title = meta.get("title")
    if not title:
        return None
    parts = [title]
    brand = meta.get("brand")
    if brand:
        parts.append(f"by {brand}")
    categories = meta.get("categories") or []
    if categories and categories[0]:
        parts.append(f"({categories[0][-1]})")
    return " ".join(parts)
