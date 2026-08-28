from __future__ import annotations
import ast
import gzip
import json
import re
from pathlib import Path

def load_id_map(path: str | Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            key, idx = line.split('\t')
            mapping[key] = int(idx)
    return mapping

def _open_maybe_gzip(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def load_train_pairs(train_json_path: str | Path) -> set[tuple[int, int]]:
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train = json.load(f)
    pairs: set[tuple[int, int]] = set()
    for uid_str, items in train.items():
        uid = int(uid_str)
        for iid in items:
            pairs.add((uid, int(iid)))
    return pairs

def load_user_histories(raw_reviews_path: str | Path, user_map: dict[str, int], item_map: dict[str, int], max_items_per_user: int=20, allowed_pairs: set[tuple[int, int]] | None=None) -> tuple[list[list[dict]], list[list[dict]]]:
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
            reviewer = obj.get('reviewerID')
            asin = obj.get('asin')
            uid = user_map.get(reviewer)
            iid = item_map.get(asin)
            if uid is None or iid is None:
                n_skipped += 1
                continue
            if allowed_pairs is not None and (uid, iid) not in allowed_pairs:
                n_skipped += 1
                continue
            rating = obj.get('overall')
            summary = (obj.get('summary') or '').strip()
            body = (obj.get('reviewText') or '').strip()
            review = (summary + '. ' + body).strip('. ').strip() if summary else body
            ts = int(obj.get('unixReviewTime') or 0)
            rec = {'item_id': iid, 'asin': asin, 'rating': float(rating) if rating is not None else None, 'review': review}
            per_user[uid].append((ts, rec))
            per_item[iid].append({'review': review, 'rating': rec['rating']})
    user_histories: list[list[dict]] = []
    for recs in per_user:
        recs.sort(key=lambda x: x[0], reverse=True)
        user_histories.append([r for _, r in recs[:max_items_per_user]])
    print(f'[data_utils] parsed {n_lines} reviews ({n_skipped} skipped), {n_users} users, {n_items} items')
    return (user_histories, per_item)

def load_item_descriptions(item_desc_path: str | Path | None, n_items: int) -> list[str | None]:
    descs: list[str | None] = [None] * n_items
    if item_desc_path is None:
        return descs
    with open(item_desc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            idx, text = line.split('\t', 1)
            i = int(idx)
            if 0 <= i < n_items:
                descs[i] = text
    return descs

def _parse_metadata_line(line: str) -> dict | None:
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
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    match = re.search('\\d+(?:[.,]\\d+)?', text.replace(',', ''))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None

def _parse_categories(obj: dict) -> list[list[str]]:
    nested = obj.get('categories')
    if nested:
        if isinstance(nested[0], (list, tuple)):
            return [list(path) for path in nested if path]
        return [[str(x) for x in nested]]
    flat = obj.get('category')
    if flat:
        if isinstance(flat, (list, tuple)):
            return [[str(x) for x in flat]]
        return [[str(flat)]]
    return []

def load_item_metadata(metadata_path: str | Path | None, item_map: dict[str, int], n_items: int) -> list[dict | None]:
    metas: list[dict | None] = [None] * n_items
    if metadata_path is None:
        return metas
    path = Path(metadata_path)
    if not path.exists():
        print(f'[data_utils] metadata file not found: {path} — skipping metadata enrichment')
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
            asin = obj.get('asin')
            iid = item_map.get(asin)
            if iid is None or not 0 <= iid < n_items:
                continue
            title = (obj.get('title') or '').strip() or None
            categories = _parse_categories(obj)
            price = _parse_price(obj.get('price'))
            metas[iid] = {'title': title, 'brand': (obj.get('brand') or '').strip() or None, 'categories': categories, 'price': price}
            n_matched += 1
            n_title += title is not None
            n_cat += bool(categories)
            n_price += price is not None
    print(f'[data_utils] matched metadata for {n_matched}/{n_items} items (title={n_title}, category={n_cat}, price={n_price})')
    if n_matched and n_cat == 0:
        print('[data_utils] WARNING: no categories parsed from any record. The metadata schema may be unrecognized, which would strip the category path from every prompt.')
    return metas

def build_item_desc_from_metadata(meta: dict | None) -> str | None:
    if not meta:
        return None
    title = meta.get('title')
    if not title:
        return None
    parts = [title]
    brand = meta.get('brand')
    if brand:
        parts.append(f'by {brand}')
    categories = meta.get('categories') or []
    if categories and categories[0]:
        parts.append(f'({categories[0][-1]})')
    return ' '.join(parts)
