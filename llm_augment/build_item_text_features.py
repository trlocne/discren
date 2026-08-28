"""
Offline pipeline to generate LLM-enriched item text features for ablation.

Outputs (under --data-dir, default data/Clothing):
- text_llm_feat.npy      (n_items, d_s), row-aligned with item_list.txt
- item_profiles.txt      item_id<TAB>generated description

Example:
    python -m llm_augment.build_item_text_features \
        --backend vllm \
        --llm-model Qwen/Qwen3-8B \
        --encoder sentence-transformers/stsb-roberta-large
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import prompts as P
from .data_utils import (
    load_id_map,
    load_item_metadata,
    load_train_pairs,
    load_user_histories,
)
from .boilerplate import format_diversity, opening_diversity, strip_all
from .encoder import build_embedder
from .llm_backend import build_backend
from .postprocess import build_postprocessor
from .provenance import write_provenance
from .validation import ValidationStats, generate_validated


# Written for items with neither catalog metadata nor a train-split review. Kept
# greppable and distinct from [GENERATION_FAILED], which means the LLM was asked
# and produced something unusable; this one means the LLM was never asked.
SENTINEL_NO_SOURCE = "[NO_SOURCE_DATA]"


def _build_item_review_block(records: list[dict], max_reviews: int = 12,
                             max_review_chars: int = 220) -> str:
    """Render item reviews into a compact block for ITEM_PROFILE_USER_PROMPT."""
    lines: list[str] = []
    for rec in records[:max_reviews]:
        rating = rec.get("rating")
        review = (rec.get("review") or "").strip().replace("\n", " ")
        if not review:
            continue
        if len(review) > max_review_chars:
            review = review[:max_review_chars].rstrip() + "..."
        prefix = f"- ({rating:g}/5) " if rating is not None else "- "
        lines.append(prefix + review)
    return "\n".join(lines) if lines else "- No customer review text available for this item."


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LLM item text features.")
    # Paths
    ap.add_argument("--data-dir", default="data/Clothing",
                    help="Folder holding item_list.txt / user_list.txt / *.npy")
    ap.add_argument("--raw-reviews",
                    default="data/Clothing_raw/reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
                    help="Raw Amazon reviews (.json or .json.gz)")
    ap.add_argument("--train-json",
                    default="data/Clothing/5-core/train.json",
                    help="Train split ({user_id:[item_id,...]}). Only reviews written as "
                         "part of a TRAIN interaction are shown to the LLM, so held-out "
                         "review text cannot leak into the item features. Set to '' to "
                         "disable filtering (NOT valid for reported results).")
    ap.add_argument("--metadata", default=None,
                    help="Optional Amazon metadata.json.gz (title/brand/categories/price). "
                         "Highly recommended: metadata is the primary source, reviews fill gaps.")
    ap.add_argument("--out-name", default="text_llm_feat.npy")
    ap.add_argument("--profile-log-name", default="item_profiles.txt")

    # LLM. No default backend on purpose -- see the note in build_user_profiles.py.
    ap.add_argument("--backend", required=True, choices=["hf", "vllm", "echo"],
                    help="Generation backend. 'echo' is a dry-run stub: it writes "
                         "placeholder text and must never be used for reported results.")
    ap.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--domain", default="clothing",
                    help="Dataset domain used to phrase the system prompt "
                         "(clothing | sports | generic). Describing Sports items to "
                         "the model as fashion invites ungrounded inference.")
    # Aligned with llm_augment/config.yaml. These used to be 120 / 0.2 here and
    # 160 / 0.3 in the config, so the CLI and the documented recipe silently
    # produced different features.
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--llm-batch-size", type=int, default=32)

    # Output validation
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Retries for generations that fail format/length validation.")
    ap.add_argument("--retry-temperature", type=float, default=0.7,
                    help="Temperature for retries; higher than the base temperature so "
                         "the model does not repeat the same malformed output.")
    ap.add_argument("--max-failure-rate", type=float, default=0.02,
                    help="Abort before writing artifacts if more than this fraction of "
                         "generations remain invalid after retries (0 = never abort).")
    ap.add_argument("--max-ungrounded-rate", type=float, default=0.60,
                    help="Abort if more than this fraction of items have neither "
                         "metadata nor a train review. A high rate almost always "
                         "means the raw files do not match item_list.txt rather "
                         "than a genuinely sparse catalogue (0 = never abort).")

    # Encoder
    ap.add_argument("--encoder", default="sentence-transformers/stsb-roberta-large")
    ap.add_argument("--encoder-batch-size", type=int, default=256)

    # Embedding quality -- see the note in build_user_profiles.py.
    #
    # OFF by default on this side, unlike the user side. Item descriptions are
    # anchored by a real catalog title, so the model opens with the product name
    # rather than a template: measured on the 23033 generated descriptions,
    # openings are already 74% distinct (top1 = 0.4%) and the pattern matched
    # only 1 text. Stripping is therefore all risk and no benefit here. The flag
    # stays available for corpora generated without metadata, where the model
    # has nothing concrete to open with and does fall back to templates.
    ap.add_argument("--strip-boilerplate", dest="strip_boilerplate",
                    action="store_true", default=False,
                    help="Strip formulaic openings ('This product is a...') before "
                         "encoding. Off by default: item openings are already "
                         "diverse because a real catalog title anchors them.")
    ap.add_argument("--no-strip-boilerplate", dest="strip_boilerplate",
                    action="store_false",
                    help="Encode the raw generated text verbatim (the default).")
    ap.add_argument("--postprocess", default="whiten",
                    choices=["none", "center", "abtt", "whiten"],
                    help="Isotropy correction applied to the encoded matrix.")
    ap.add_argument("--postprocess-alpha", type=float, default=1.0,
                    help="Whitening strength (1.0 = full).")
    ap.add_argument("--postprocess-components", type=int, default=1,
                    help="Directions removed when --postprocess=abtt.")

    # Behaviour
    ap.add_argument("--max-reviews-per-item", type=int, default=12)
    ap.add_argument("--gen-batch-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process first N items (0 = all). For testing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    user_map = load_id_map(data_dir / "user_list.txt")
    item_map = load_id_map(data_dir / "item_list.txt")
    n_items = max(item_map.values()) + 1
    print(f"[build-item] items={n_items}")

    # Restrict to TRAIN interactions. Without this the LLM reads review text
    # from the val/test split and writes it into text_llm_feat.npy, which is a
    # feature used at evaluation time -- held-out signal would leak straight
    # into the reported metrics. The user-side builder has always filtered;
    # this side did not, which made every item-LLM number unsafe to report.
    allowed_pairs = None
    if args.train_json:
        allowed_pairs = load_train_pairs(args.train_json)
        print(f"[build-item] restricting to {len(allowed_pairs)} train interactions "
              f"(from {args.train_json})")
    else:
        print("[build-item] WARNING: --train-json is empty, so val/test review text "
              "WILL leak into the generated item features. Do not report results "
              "produced this way.")

    # We only need per-item reviews. user_histories is intentionally ignored.
    _user_histories, item_reviews = load_user_histories(
        args.raw_reviews,
        user_map=user_map,
        item_map=item_map,
        max_items_per_user=1,
        allowed_pairs=allowed_pairs,
    )
    item_metas = load_item_metadata(args.metadata, item_map=item_map, n_items=n_items)

    n_target = n_items if args.limit <= 0 else min(args.limit, n_items)

    # An item with no metadata AND no train review gives the model nothing to
    # describe, so anything it writes is invention -- and invention is invisible
    # in the output, because a fabricated product description reads exactly like
    # a grounded one. Those items are therefore not sent to the LLM at all; they
    # receive an explicit sentinel and a zero feature row, which the model treats
    # as "no LLM signal for this item" rather than as a confident wrong claim.
    # On the 2018 Sports dump this is a large fraction of the catalogue, which is
    # why the check is enforced rather than merely reported.
    prompts: list[str] = []
    prompt_index: list[int] = []      # position in `prompts` -> item id
    ungrounded: list[int] = []
    for iid in range(n_target):
        has_meta = bool(item_metas[iid] and (
            (item_metas[iid].get("title") or "").strip()
            or item_metas[iid].get("categories")
        ))
        has_review = any(
            (rec.get("review") or "").strip() for rec in item_reviews[iid]
        )
        if not (has_meta or has_review):
            ungrounded.append(iid)
            continue
        meta_block = P.build_item_metadata_block(item_metas[iid])
        review_block = _build_item_review_block(
            item_reviews[iid],
            max_reviews=args.max_reviews_per_item,
        )
        prompts.append(P.ITEM_PROFILE_USER_PROMPT.format(
            metadata_block=meta_block,
            review_block=review_block,
        ))
        prompt_index.append(iid)

    n_grounded = len(prompts)
    ungrounded_rate = len(ungrounded) / max(n_target, 1)
    print(f"[build-item] source coverage: {n_grounded}/{n_target} items have "
          f"metadata or a train review ({100.0 * (1 - ungrounded_rate):.1f}%)")
    if ungrounded:
        print(f"[build-item] {len(ungrounded)} items "
              f"({100.0 * ungrounded_rate:.1f}%) have NEITHER source and will be "
              f"marked {SENTINEL_NO_SOURCE} with a zero feature row "
              f"(not sent to the LLM).")
    if args.max_ungrounded_rate > 0 and ungrounded_rate > args.max_ungrounded_rate:
        raise SystemExit(
            f"[build-item] ABORT: {100.0 * ungrounded_rate:.1f}% of items have no "
            f"metadata and no train review, above the --max-ungrounded-rate of "
            f"{100.0 * args.max_ungrounded_rate:.1f}%. The raw files probably do "
            f"not match this benchmark's item_list.txt. No artifacts were written."
        )

    backend_kwargs = dict(
        model_name=args.llm_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    if args.backend == "hf":
        backend_kwargs["batch_size"] = args.llm_batch_size
    if args.backend == "echo":
        backend_kwargs = {}
    llm = build_backend(args.backend, **backend_kwargs)

    domain_phrase = P.resolve_domain(args.domain)
    item_system_prompt = P.ITEM_PROFILE_SYSTEM_PROMPT.format(domain=domain_phrase)
    print(f"[build-item] domain: {domain_phrase}")
    print(f"[build-item] generating {n_grounded} item descriptions with {args.backend} ...")
    t0 = time.time()
    generated: list[str] = []
    stats = ValidationStats()
    for start in range(0, n_grounded, args.gen_batch_size):
        chunk = prompts[start : start + args.gen_batch_size]
        batch, stats = generate_validated(
            llm,
            item_system_prompt,
            chunk,
            kind="item",
            max_retries=args.max_retries,
            retry_temperature=args.retry_temperature,
            stats=stats,
        )
        generated.extend(batch)
        done = min(start + args.gen_batch_size, n_grounded)
        print(f"  {done}/{n_grounded} ({(time.time() - t0):.0f}s)")

    # Scatter the generated text back to item-id order, leaving ungrounded items
    # at the sentinel so the row order still matches item_list.txt.
    descriptions: list[str] = [SENTINEL_NO_SOURCE] * n_target
    for pos, iid in enumerate(prompt_index):
        descriptions[iid] = generated[pos]
    assert len(descriptions) == n_target, (len(descriptions), n_target)
    print(stats.summary())
    failure_rate = stats.failed / max(stats.total, 1)
    if stats.failed:
        print(f"[build-item] WARNING: {stats.failed} description(s) "
              f"({100.0 * failure_rate:.2f}%) could not be repaired and are marked "
              f"[GENERATION_FAILED] in the audit log.")
    # Fail loudly rather than silently shipping a degraded feature matrix.
    if args.max_failure_rate > 0 and failure_rate > args.max_failure_rate:
        raise SystemExit(
            f"[build-item] ABORT: {100.0 * failure_rate:.2f}% of generations are "
            f"unusable, above the --max-failure-rate of "
            f"{100.0 * args.max_failure_rate:.2f}%. No artifacts were written. "
            f"Check the model/prompt, or raise the threshold deliberately."
        )

    log_path = data_dir / args.profile_log_name
    with open(log_path, "w", encoding="utf-8") as f:
        for iid, txt in enumerate(descriptions):
            f.write(f"{iid}\t{txt.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print(f"[build-item] wrote descriptions -> {log_path}")

    # Only what the ENCODER sees is stripped; the audit log above is verbatim.
    to_encode = descriptions
    if args.strip_boilerplate:
        before = opening_diversity(descriptions)
        to_encode, n_stripped = strip_all(descriptions, kind="item")
        after = opening_diversity(to_encode)
        print(f"[build-item] boilerplate: stripped {n_stripped}/{len(descriptions)} openings")
        print(f"[build-item]   before  {format_diversity(before)}")
        print(f"[build-item]   after   {format_diversity(after)}")

    if args.encoder == "hashing":
        embedder = build_embedder("hashing")
    else:
        embedder = build_embedder(args.encoder, batch_size=args.encoder_batch_size)

    print(f"[build-item] encoding descriptions with {args.encoder} (dim={embedder.dim}) ...")
    emb = embedder.encode(to_encode)

    if n_target < n_items:
        pad = np.zeros((n_items - n_target, emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=0)

    # Zero the ungrounded rows. Encoding the sentinel string would give every one
    # of them the *same* non-zero vector, which the kNN step would then read as a
    # tight cluster of mutually similar items -- a structure invented entirely by
    # the absence of data. A zero row instead means "no LLM signal here", is
    # excluded from the whitening fit below, and stays zero through it.
    for iid in ungrounded:
        if iid < emb.shape[0]:
            emb[iid] = 0.0

    # Remove the shared cone the encoder puts every sentence in. Fitted on the
    # generated rows only; zero rows are excluded and stay zero.
    post = build_postprocessor(
        method=args.postprocess,
        n_components=args.postprocess_components,
        alpha=args.postprocess_alpha,
    )
    if args.postprocess != "none":
        emb = post.fit_apply(emb)
        print(f"[build-item] {post.describe()} applied")

    out_path = data_dir / args.out_name
    emb = emb.astype(np.float32)
    np.save(out_path, emb)
    print(f"[build-item] saved {emb.shape} -> {out_path}")

    meta_file = write_provenance(
        out_path,
        emb,
        kind="item_text",
        backend=args.backend,
        llm_model=args.llm_model,
        encoder=args.encoder,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        n_generated=n_target,
        train_filtered=bool(args.train_json),
        validation={
            "total": stats.total,
            "cleaned": stats.cleaned,
            "retried": stats.retried,
            "recovered": stats.recovered,
            "failed": stats.failed,
            "reasons": stats.reasons,
        },
        extra={
            "max_reviews_per_item": args.max_reviews_per_item,
            "strip_boilerplate": bool(args.strip_boilerplate),
            "postprocess": post.describe(),
            "domain": args.domain,
            "source_coverage": {
                "grounded": n_grounded,
                "ungrounded": len(ungrounded),
                "ungrounded_rate": round(ungrounded_rate, 4),
            },
        },
    )
    print(f"[build-item] wrote provenance -> {meta_file}")
    print("[build-item] done. Row i aligns with item id i from item_list.txt.")


if __name__ == "__main__":
    main()
