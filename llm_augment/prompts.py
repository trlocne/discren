from __future__ import annotations
from textwrap import dedent
DOMAINS = {'clothing': 'a fashion marketplace (clothing, shoes and jewelry)', 'sports': 'a sports and outdoors marketplace (fitness equipment, outdoor gear, apparel and accessories)', 'generic': 'an online retail marketplace'}
DEFAULT_DOMAIN = 'clothing'

def resolve_domain(name: str | None) -> str:
    if not name:
        return DOMAINS[DEFAULT_DOMAIN]
    return DOMAINS.get(str(name).strip().lower(), DOMAINS['generic'])
USER_PROFILE_SYSTEM_PROMPT = dedent('    You are an expert e-commerce preference analyst. Given the purchase and\n    review history of a single anonymous shopper on {domain}, you summarise WHO\n    this shopper is as a consumer — in a way that would let someone tell this\n    shopper apart from a DIFFERENT shopper with a similar-looking history.\n\n    Grounding rules (avoid generic filler):\n    - Every claim must be traceable to the history: if a brand, category,\n      colour, material or price tier appears more than once, name it\n      explicitly (e.g. "repeatedly buys Nike running shoes", not just\n      "prefers athletic wear"). Prefer concrete nouns over abstract\n      adjectives whenever the history supports them.\n    - Do NOT default to filler phrases like "mid-range priced", "casual and\n      versatile", "values comfort and quality" unless the history gives no\n      more specific signal at all — these phrases are true of most shoppers\n      and add no distinguishing information. If you catch yourself about to\n      write one, look again for a more specific detail (a repeated brand, an\n      unusual category mix, a distinctive complaint or compliment) instead.\n    - Lead with whatever is MOST DISTINCTIVE about this particular shopper\n      (an unusual combination of categories, a strong like/dislike, a clear\n      recipient other than themselves, a narrow niche), not with a generic\n      opening sentence. Different shoppers should read differently even if\n      both happen to shop mid-range casual wear.\n    - Infer sentiment from the reviews (what they like vs. dislike) and state\n      it concretely (what specifically worked or failed), not just as\n      "positive/negative sentiment".\n    - Do NOT list individual product ids, do NOT invent facts not supported by\n      the history, do NOT add greetings, headers, markdown or bullet points.\n\n    Format:\n    - ONE dense paragraph, 40-80 words, third person ("This shopper ...").\n    - Vary sentence structure and word choice between profiles — do not reuse\n      the same opening template ("This shopper prefers...") for every writeup;\n      start instead with whatever detail is most distinctive for THIS shopper.\n    - Output ONLY the profile paragraph, nothing else.\n    ')
USER_PROFILE_USER_PROMPT = dedent("    Below is the interaction history of one shopper. Each line is an item they\n    interacted with, optionally followed by the rating they gave and a short\n    excerpt of their review.\n\n    ---\n    {history_block}\n    ---\n\n    First, silently identify: (a) any brand, category, colour or material that\n    repeats across multiple lines, (b) the most distinctive or unusual thing\n    about this shopper compared to a typical shopper, and (c) any concrete\n    likes/dislikes stated in the reviews. Then write the shopper's preference\n    profile now, grounded in those specifics rather than generic descriptors.\n    ")
ITEM_PROFILE_SYSTEM_PROMPT = dedent('    You are a catalog content specialist preparing product text for the\n    RECOMMENDER SYSTEM of {domain}. Your output will be embedded into\n    a vector and used for item-item similarity and retrieval, so it must be\n    dense with the concrete, discriminative attributes a recommendation model\n    can key on: product type, category/subcategory, intended user (men /\n    women / kids / unisex), style, color, material, fit or sizing,\n    occasion/use-case, and price tier.\n\n    You are given structured catalog metadata (title, brand, category path,\n    price) and, optionally, a few customer reviews for the SAME product.\n\n    Rules:\n    - ONE paragraph, 30-70 words, third person, no marketing hype, no\n      first/second person, no emojis, no markdown.\n    - Treat metadata (title, brand, category, price) as ground truth. Use\n      reviews only to add attributes not already covered by metadata (e.g.\n      fit, comfort, durability, sizing accuracy) — never contradict metadata.\n    - If reviews conflict with each other, prefer the majority sentiment.\n    - Mention brand and category naturally if present; mention price tier\n      qualitatively (budget / mid-range / premium) instead of the raw number.\n    - Do NOT invent attributes not supported by the given metadata or reviews.\n    - Output ONLY the description paragraph, nothing else.\n    ')
ITEM_PROFILE_USER_PROMPT = dedent('    Catalog metadata for this product:\n    ---\n    {metadata_block}\n    ---\n\n    Customer reviews for this product (may be empty):\n    ---\n    {review_block}\n    ---\n\n    Write the recommender-system product description now.\n    ')

def build_item_metadata_block(meta: dict | None) -> str:
    if not meta:
        return '(no catalog metadata available for this product)'
    lines: list[str] = []
    title = meta.get('title')
    if title:
        lines.append(f'Title: {title}')
    brand = meta.get('brand')
    if brand:
        lines.append(f'Brand: {brand}')
    categories = meta.get('categories') or []
    if categories:
        flat_paths = [' > '.join(path) for path in categories if path]
        if flat_paths:
            lines.append(f"Category: {'; '.join(flat_paths)}")
    price = meta.get('price')
    if price is not None:
        lines.append(f'Price: ${price:.2f}')
    return '\n'.join(lines) if lines else '(no catalog metadata available for this product)'

def build_user_history_block(records: list[dict], max_items: int=20, max_review_chars: int=160) -> str:
    lines: list[str] = []
    for rec in records[:max_items]:
        desc = (rec.get('item_desc') or f"item {rec.get('item_id', '?')}").strip()
        parts = [f'- {desc}']
        rating = rec.get('rating')
        if rating is not None:
            parts.append(f'(rated {rating:g}/5)')
        review = (rec.get('review') or '').strip().replace('\n', ' ')
        if review:
            if len(review) > max_review_chars:
                review = review[:max_review_chars].rstrip() + '…'
            parts.append(f'— "{review}"')
        lines.append(' '.join(parts))
    return '\n'.join(lines) if lines else '- (no recorded interactions)'
