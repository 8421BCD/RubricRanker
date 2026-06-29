#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Roll out the RubricRanker model on the verl training set and score every
rollout with the set-selection reward function.

For each example in train.jsonl:
  1. Take the chat-format `prompt` field (system + user).
  2. Render it with the model's tokenizer in *non-thinking* mode
     (`enable_thinking=False`) using `apply_chat_template`.
  3. Generate `n_rollouts` (default 8) samples per prompt with vLLM, using
     Qwen-style sampling (T=0.7, top_p=0.8, top_k=20, min_p=0).
  4. Score every rollout with
     `verl.utils.reward_score.set_selection.compute_score`. All rollouts of
     the current batch are scored concurrently with a thread pool because the
     scoring function makes blocking HTTP calls to the GPT eval API.
  5. Append one JSON line per example with rollouts / rewards / mean / std.

The output is appended after every batch, and on restart already-finished
queries are skipped, so the script is safely resumable.
"""

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# Make `verl.utils.reward_score.set_selection` importable when this script is
# run from the verl repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from verl.utils.reward_score.set_selection import compute_score  # noqa: E402


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_done_queries(path: str) -> set:
    """Collect `query` of already-completed rows for resume."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("query")
            if q is not None:
                done.add(q)
    return done


# --------------------------------------------------------------------------- #
# Scoring (thread-pool over compute_score, which is HTTP-bound)
# --------------------------------------------------------------------------- #

def score_one(rollout_text: str, extra_info: Dict[str, Any]) -> float:
    try:
        r = compute_score(rollout_text, ground_truth=None, extra_info=extra_info)
        return float(r)
    except Exception as e:  # noqa: BLE001
        print(f"[score_one] error: {type(e).__name__}: {e}", flush=True)
        return -1.0


def score_batch_concurrent(
    rollouts_per_example: List[List[str]],
    extra_infos: List[Dict[str, Any]],
    workers: int,
) -> List[List[float]]:
    """
    rollouts_per_example[i] is a list of N rollouts for example i.
    Returns rewards in the same shape.
    """
    # Flatten with bookkeeping.
    flat_jobs = []  # list of (example_idx, rollout_idx, text, extra_info)
    for ei, rolls in enumerate(rollouts_per_example):
        for ri, text in enumerate(rolls):
            flat_jobs.append((ei, ri, text, extra_infos[ei]))

    rewards: List[List[float]] = [
        [0.0] * len(rolls) for rolls in rollouts_per_example
    ]

    if not flat_jobs:
        return rewards

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        future_to_loc = {
            ex.submit(score_one, text, info): (ei, ri)
            for (ei, ri, text, info) in flat_jobs
        }
        for fut in as_completed(future_to_loc):
            ei, ri = future_to_loc[fut]
            rewards[ei][ri] = fut.result()
    return rewards


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/apdcephfs_zwfy4/share_303945702/wenhanliu/searchrubrics/verl/data/train.jsonl",
    )
    ap.add_argument(
        "--output",
        default="/apdcephfs_zwfy4/share_303945702/wenhanliu/searchrubrics/verl/data/train_rollouts_scores.jsonl",
    )
    ap.add_argument(
        "--model",
        default="/apdcephfs_zwfy4/share_303945702/wenhanliu/trained_models/rubricranker_sft",
    )
    ap.add_argument("--n_rollouts", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="number of prompts fed to vLLM per batch")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--max_model_len", type=int, default=None,
                    help="optional cap; default = model config")
    ap.add_argument("--scoring_workers", type=int, default=32,
                    help="thread pool size for compute_score (HTTP-bound)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process first N (post-resume) examples; for debug")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    # --- load tokenizer + vLLM (lazy import so we keep startup logs clean) ---
    print(f"[init] loading tokenizer from {args.model}", flush=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[init] loading vLLM model from {args.model}", flush=True)
    from vllm import LLM, SamplingParams
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # --- resume support ---
    done = load_done_queries(args.output)
    if done:
        print(f"[resume] {len(done)} examples already in {args.output}, skipping",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    # --- streaming batch loop ---
    pending: List[Dict[str, Any]] = []
    n_processed = 0
    n_seen = 0
    t0 = time.time()

    def flush_batch(batch: List[Dict[str, Any]]):
        nonlocal n_processed
        if not batch:
            return

        # 1) build chat-template prompts
        rendered_prompts: List[str] = []
        for ex in batch:
            text = tokenizer.apply_chat_template(
                ex["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rendered_prompts.append(text)

        # 2) generate N rollouts per prompt in one vLLM call
        gen_t0 = time.time()
        outputs = llm.generate(rendered_prompts, sampling_params)
        gen_t = time.time() - gen_t0

        # vLLM preserves input order; line them up.
        rollouts_per_example: List[List[str]] = []
        for out in outputs:
            texts = [o.text for o in out.outputs]
            # safety pad if model emits fewer than n_rollouts (shouldn't happen)
            while len(texts) < args.n_rollouts:
                texts.append("")
            rollouts_per_example.append(texts[: args.n_rollouts])

        # 3) build extra_info per example for compute_score
        extra_infos: List[Dict[str, Any]] = []
        for ex in batch:
            extra_infos.append({
                "initial_list": ex.get("initial_list") or [],
                "rubrics": ex.get("rubrics") or {},
                "scenario": ex.get("scenario") or "rag",
                "query": ex.get("query") or "",
                "thinking": ex.get("thinking") or "",
            })

        # 4) score all rollouts of this batch concurrently
        score_t0 = time.time()
        rewards = score_batch_concurrent(
            rollouts_per_example, extra_infos, workers=args.scoring_workers
        )
        score_t = time.time() - score_t0

        # 5) write one row per example
        for ex, rolls, rews in zip(batch, rollouts_per_example, rewards):
            mean = float(sum(rews) / len(rews)) if rews else 0.0
            std = float(statistics.pstdev(rews)) if len(rews) > 1 else 0.0
            row = {
                "dataset": ex.get("dataset"),
                "scenario": ex.get("scenario"),
                "query": ex.get("query"),
                "rollouts": rolls,
                "rewards": rews,
                "reward_mean": mean,
                "reward_std": std,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()

        n_processed += len(batch)
        elapsed = time.time() - t0
        print(
            f"[batch] size={len(batch)} gen={gen_t:.1f}s score={score_t:.1f}s "
            f"| done={n_processed} elapsed={elapsed:.1f}s "
            f"avg={elapsed / max(1, n_processed):.2f}s/ex",
            flush=True,
        )

    try:
        for ex in iter_jsonl(args.input):
            n_seen += 1
            if ex.get("query") in done:
                continue
            if not isinstance(ex.get("prompt"), list) or not ex["prompt"]:
                print(f"[skip] no chat-format prompt: query={ex.get('query')!r}",
                      flush=True)
                continue
            pending.append(ex)

            # honor --limit (post-resume)
            if args.limit is not None and (n_processed + len(pending)) >= args.limit:
                # cap exactly at the limit
                room = args.limit - n_processed
                pending = pending[:room]
                flush_batch(pending)
                pending = []
                break

            if len(pending) >= args.batch_size:
                flush_batch(pending)
                pending = []

        if pending:
            flush_batch(pending)
    finally:
        out_f.close()

    print(f"[done] seen={n_seen} processed={n_processed} "
          f"output={args.output} total={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
