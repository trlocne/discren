"""
Offline LLM Semantic-Profiling pipeline (paper module A2).

End-to-end flow
---------------
1. Load ``user_list.txt`` / ``item_list.txt`` id maps.
2. Parse raw Amazon reviews and group them per user (row-aligned with
   ``user_list.txt``).
3. For each user, render the default prompt and call an OPEN-SOURCE LLM to
   produce a short natural-language preference profile.
4. Encode every profile with a sentence encoder.
5. Save::

       data/Clothing/user_profile_feat.npy   (n_users, d_s)   float32
       data/Clothing/user_profiles.txt       user_id<TAB>profile   (audit log)

   ``user_profile_feat.npy`` is row-aligned with ``user_list.txt`` exactly the
   way ``text_feat.npy`` is row-aligned with ``item_list.txt``.

Example
-------
Dry run (no GPU, sanity-check alignment + I/O)::

    python -m llm_augment.build_user_profiles --backend echo --limit 100

Real run with a local open-source model via vLLM::

    python -m llm_augment.build_user_profiles \\
        --backend vllm \\
        --llm-model Qwen/Qwen3-8B \\
        --encoder sentence-transformers/stsb-roberta-large

Every run also writes ``<out_name>.meta.json`` recording the backend, model,
temperature, prompt version, encoder, content hash and validation tally.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import prompts as P
from .data_utils import (
    build_item_desc_from_metadata,
    load_id_map,
    load_item_descriptions,
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build LLM user semantic profiles.")
    # Paths
    ap.add_argument("--data-dir", default="data/Clothing",
                    help="Folder holding item_list.txt / user_list.txt / *.npy")
    ap.add_argument("--raw-reviews",
                    default="data/Clothing_raw/reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
                    help="Raw Amazon 5-core reviews (.json or .json.gz)")
    ap.add_argument("--train-json",
                    default="data/Clothing/5-core/train.json",
                    help="Train split ({user_id:[item_id,...]}). Only these interactions "
                         "are used for profiling to avoid val/test leakage. "
                         "Set to '' to disable filtering (use ALL reviews).")
    ap.add_argument("--item-desc", default=None,
                    help="Optional item_id<TAB>description file for richer prompts "
                         "(lowest priority — overridden by --item-profiles-log / --metadata)")
    ap.add_argument("--metadata", default=None,
                    help="Optional Amazon metadata.json.gz (title/brand/categories). "
                         "Used to name items in the shopper history instead of bare ids.")
    ap.add_argument("--item-profiles-log", default=None,
                    help="Optional item_profiles.txt from build_item_text_features.py "
                         "(LLM-generated item descriptions). Used ONLY as a gap-filler "
                         "for items with no catalog metadata — see --trust-llm-item-desc.")
    ap.add_argument("--trust-llm-item-desc", action="store_true",
                    help="Let LLM-generated item descriptions OVERRIDE real catalog "
                         "metadata in the history block. Off by default: doing so makes "
                         "one LLM's hallucinations the input to the next LLM stage, so a "
                         "wrong item description silently becomes a wrong user profile. "
                         "Only enable if you have separately validated item_profiles.txt.")
    ap.add_argument("--out-name", default="user_profile_feat.npy")
    # Configurable for the same reason --out-name is: a dry run must be able to
    # keep ALL of its outputs away from the real ones. This filename used to be
    # hardcoded, so a --limit 200 smoke test truncated the real 39k-row audit
    # log even though its .npy went somewhere safe.
    ap.add_argument("--profile-log-name", default="user_profiles.txt",
                    help="Audit log of generated profile text (user_id<TAB>profile).")

    # LLM. There is deliberately NO default backend: 'echo' used to be the
    # default, so a run that simply forgot --backend wrote stub text to the very
    # same artifact names a real run uses, and nothing downstream could tell the
    # difference. The backend must now be stated explicitly.
    ap.add_argument("--backend", required=True, choices=["hf", "vllm", "echo"],
                    help="Generation backend. 'echo' is a dry-run stub: it writes "
                         "placeholder text and must never be used for reported results.")
    ap.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--domain", default="clothing",
                    help="Dataset domain used to phrase the system prompt "
                         "(clothing | sports | generic).")
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

    # Encoder
    ap.add_argument("--encoder", default="sentence-transformers/stsb-roberta-large")
    ap.add_argument("--encoder-batch-size", type=int, default=256)

    # Embedding quality. Sentence encoders emit a narrow cone -- two random
    # profiles here had cosine 0.64 -- and the injection block cannot undo that,
    # since it normalizes the projection rather than the input.
    ap.add_argument("--strip-boilerplate", dest="strip_boilerplate",
                    action="store_true", default=True,
                    help="Strip formulaic openings ('This shopper consistently "
                         "purchases...') before encoding. The audit log keeps the "
                         "original text. On by default.")
    ap.add_argument("--no-strip-boilerplate", dest="strip_boilerplate",
                    action="store_false",
                    help="Encode the raw generated text verbatim.")
    ap.add_argument("--postprocess", default="whiten",
                    choices=["none", "center", "abtt", "whiten"],
                    help="Isotropy correction applied to the encoded matrix.")
    ap.add_argument("--postprocess-alpha", type=float, default=1.0,
                    help="Whitening strength (1.0 = full).")
    ap.add_argument("--postprocess-components", type=int, default=1,
                    help="Directions removed when --postprocess=abtt.")

    # Behaviour
    ap.add_argument("--max-items-per-user", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N users (0 = all). For testing.")
    ap.add_argument("--gen-batch-size", type=int, default=1024,
                    help="How many user prompts to send to the LLM per call.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1. id maps ────────────────────────────────────────────────────────────
    user_map = load_id_map(data_dir / "user_list.txt")
    item_map = load_id_map(data_dir / "item_list.txt")
    n_users = max(user_map.values()) + 1
    n_items = max(item_map.values()) + 1
    print(f"[build] {n_users} users, {n_items} items")

    # 2. per-user history (TRAIN-only to avoid val/test leakage) ──────────────
    allowed_pairs = None
    if args.train_json:
        allowed_pairs = load_train_pairs(args.train_json)
        print(f"[build] restricting to {len(allowed_pairs)} train interactions "
              f"(from {args.train_json})")
    user_histories, _item_reviews = load_user_histories(
        args.raw_reviews, user_map, item_map,
        max_items_per_user=args.max_items_per_user,
        allowed_pairs=allowed_pairs,
    )

    # Item descriptions for the history block, lowest -> highest priority:
    #   1. bare item id (implicit fallback in build_user_history_block)
    #   2. --item-desc flat file (legacy)
    #   3. --item-profiles-log LLM-generated description (gap-filler only)
    #   4. --metadata catalog title (REAL product name — authoritative)
    #
    # Ordering rationale. An earlier version put the LLM-generated description
    # last, so it overrode real catalog metadata. That turns the two LLM stages
    # into a chain: whatever the item stage confabulates about a product becomes
    # the *only* thing the user stage ever sees about it, and the error is
    # laundered into user_profile_feat.npy with no way to detect it downstream.
    # Observed metadata beats generated text, so metadata is applied last and
    # wins. The LLM text still fills the (many) items with no metadata match,
    # which is where it actually adds information.
    item_descs = load_item_descriptions(args.item_desc, n_items)

    n_llm_desc = 0
    if args.item_profiles_log:
        llm_descs = load_item_descriptions(args.item_profiles_log, n_items)
        for iid, d in enumerate(llm_descs):
            if d:
                item_descs[iid] = d
                n_llm_desc += 1

    n_meta_desc = 0
    if args.metadata:
        item_metas = load_item_metadata(args.metadata, item_map=item_map, n_items=n_items)
        for iid, meta in enumerate(item_metas):
            d = build_item_desc_from_metadata(meta)
            if not d:
                continue
            # Real metadata overwrites the generated description unless the
            # caller explicitly opts into trusting the LLM more than the catalog.
            if item_descs[iid] is None or not args.trust_llm_item_desc:
                item_descs[iid] = d
                n_meta_desc += 1

    if args.metadata or args.item_profiles_log:
        n_grounded = sum(1 for d in item_descs if d)
        print(f"[build] item descriptions: {n_grounded}/{n_items} resolved "
              f"({n_meta_desc} from catalog metadata, {n_llm_desc} LLM-generated before "
              f"metadata override)")
        if args.trust_llm_item_desc:
            print("[build] WARNING: --trust-llm-item-desc is ON. LLM-generated item text "
                  "overrides real catalog metadata, so item-stage hallucinations will "
                  "propagate into the user profiles.")

    # attach descriptions to each record
    for recs in user_histories:
        for r in recs:
            d = item_descs[r["item_id"]]
            if d:
                r["item_desc"] = d

    # 3. build prompts ─────────────────────────────────────────────────────────
    n_target = n_users if args.limit <= 0 else min(args.limit, n_users)
    user_prompts: list[str] = []
    for uid in range(n_target):
        hist = P.build_user_history_block(
            user_histories[uid], max_items=args.max_items_per_user
        )
        user_prompts.append(P.USER_PROFILE_USER_PROMPT.format(history_block=hist))

    # 4. run LLM ────────────────────────────────────────────────────────────────
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
    user_system_prompt = P.USER_PROFILE_SYSTEM_PROMPT.format(domain=domain_phrase)
    print(f"[build] domain: {domain_phrase}")
    print(f"[build] generating {n_target} profiles with backend={args.backend} …")
    t0 = time.time()
    profiles: list[str] = []
    stats = ValidationStats()
    for start in range(0, n_target, args.gen_batch_size):
        chunk = user_prompts[start : start + args.gen_batch_size]
        batch, stats = generate_validated(
            llm,
            user_system_prompt,
            chunk,
            kind="user",
            max_retries=args.max_retries,
            retry_temperature=args.retry_temperature,
            stats=stats,
        )
        profiles.extend(batch)
        done = min(start + args.gen_batch_size, n_target)
        print(f"  {done}/{n_target} ({(time.time()-t0):.0f}s)")
    assert len(profiles) == n_target, (len(profiles), n_target)
    print(stats.summary())
    failure_rate = stats.failed / max(stats.total, 1)
    if stats.failed:
        print(f"[build] WARNING: {stats.failed} profile(s) "
              f"({100.0 * failure_rate:.2f}%) could not be repaired and are marked "
              f"[GENERATION_FAILED] in the audit log.")
    # Fail loudly rather than silently shipping a degraded feature matrix.
    if args.max_failure_rate > 0 and failure_rate > args.max_failure_rate:
        raise SystemExit(
            f"[build] ABORT: {100.0 * failure_rate:.2f}% of generations are unusable, "
            f"above the --max-failure-rate of {100.0 * args.max_failure_rate:.2f}%. "
            f"No artifacts were written. Check the model/prompt, or raise the "
            f"threshold deliberately."
        )

    # audit log
    log_path = data_dir / args.profile_log_name
    with open(log_path, "w", encoding="utf-8") as f:
        for uid, prof in enumerate(profiles):
            f.write(f"{uid}\t{prof.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print(f"[build] wrote profile text → {log_path}")

    # 5. encode ────────────────────────────────────────────────────────────────
    # The audit log above keeps the original text; only what the ENCODER sees is
    # stripped. Two openings covered 66% of this corpus, so the scaffold was a
    # large, perfectly-shared component of every embedding.
    to_encode = profiles
    if args.strip_boilerplate:
        before = opening_diversity(profiles)
        to_encode, n_stripped = strip_all(profiles, kind="user")
        after = opening_diversity(to_encode)
        print(f"[build] boilerplate: stripped {n_stripped}/{len(profiles)} openings")
        print(f"[build]   before  {format_diversity(before)}")
        print(f"[build]   after   {format_diversity(after)}")

    if args.encoder == "hashing":
        embedder = build_embedder("hashing")
    else:
        embedder = build_embedder(
            args.encoder, batch_size=args.encoder_batch_size
        )
    print(f"[build] encoding profiles with {args.encoder} (dim={embedder.dim}) …")
    emb = embedder.encode(to_encode)  # (n_target, d_s)

    # If we only processed a subset, pad the rest with zeros so the file still
    # has one row per user (matches text_feat.npy convention).
    if n_target < n_users:
        pad = np.zeros((n_users - n_target, emb.shape[1]), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=0)

    # Remove the shared cone the encoder puts every sentence in. Fitted on the
    # generated rows only; the zero padding above is excluded and stays zero.
    post = build_postprocessor(
        method=args.postprocess,
        n_components=args.postprocess_components,
        alpha=args.postprocess_alpha,
    )
    if args.postprocess != "none":
        emb = post.fit_apply(emb)
        print(f"[build] {post.describe()} applied")

    out_path = data_dir / args.out_name
    emb = emb.astype(np.float32)
    np.save(out_path, emb)
    print(f"[build] saved {emb.shape} → {out_path}")

    meta_file = write_provenance(
        out_path,
        emb,
        kind="user_profile",
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
            "max_items_per_user": args.max_items_per_user,
            "trust_llm_item_desc": bool(args.trust_llm_item_desc),
            "strip_boilerplate": bool(args.strip_boilerplate),
            "postprocess": post.describe(),
            "domain": args.domain,
        },
    )
    print(f"[build] wrote provenance → {meta_file}")
    print("[build] done. This file is row-aligned with user_list.txt "
          "(like text_feat.npy is with item_list.txt).")


if __name__ == "__main__":
    main()
