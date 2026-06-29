# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# RubricRanker (set-selection) reward.
#
# Reward pipeline for one rollout:
#   1. Parse the model output (e.g. "[1] [4] [7]") into selected doc indices.
#      - If parsing entirely fails -> reward = -1.
#      - Out-of-range ids are dropped; duplicates are removed (keep-first order).
#      - If after filtering the id set is empty -> reward = 0.
#   2. Aggregate rubric scores (each in [0, 1]) weighted by their rubric weights:
#        a) set-level rubrics (Relevance / Conciseness / Consistency) are scored
#           by GPT-5.1 in ONE call (prompt lists all set-level rubrics so the
#           model outputs scores in fixed order). Score / 10 -> [0, 1].
#           If the LLM call keeps failing after retries, set-level rubrics are
#           dropped from the weighted aggregation.
#        b) doc-level rubrics (Source Authority / Timeliness) are pre-scored on
#           the candidate docs (in `extra_info["initial_list"]`). For the
#           selected doc subset, take the mean of the pre-scored values,
#           ignoring None. If all selected docs are None for that rubric,
#           drop this rubric from aggregation.
#   3. final reward = sum(weight * score) / sum(weight) over rubrics that
#      produced a value. If sum(weight) == 0 (everything dropped) -> 0.
# =============================================================================

import logging
import os
import re

import yaml
from dotenv import load_dotenv
from openai import OpenAI


logger = logging.getLogger(__name__)

# ---------------- paths / constants ----------------

_THIS_FILE = os.path.abspath(__file__)
# verl/verl/utils/reward_score/set_selection.py -> .../searchrubrics_github
_SEARCHRUBRICS_ROOT = os.path.abspath(
    os.path.join(_THIS_FILE, "..", "..", "..", "..", "..")
)
_DATA_PROCESS_DIR = os.path.join(_SEARCHRUBRICS_ROOT, "data_process")
_PROMPT_YAML_PATH = os.path.join(
    _DATA_PROCESS_DIR, "prompts", "set_level_rubric_scoring.yaml"
)
# OpenAI API key is read from dr-tulu agent's .env file.
_DOTENV_PATH = os.path.join(
    _SEARCHRUBRICS_ROOT, "dr-tulu", "agent", ".env"
)

# Load API credentials from dr-tulu/agent/.env (absolute path).
load_dotenv(dotenv_path=_DOTENV_PATH)

# OpenAI API config.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional override
HTTP_TIMEOUT = 60
MAX_ATTEMPTS = 4  # retry budget for the set-level LLM call

_OPENAI_CLIENT = None


# Truncate doc content fed into the prompt (chars per doc).
# DOC_CONTENT_MAX_CHARS = 4000

SET_LEVEL_RUBRIC_NAMES = ["Relevance", "Conciseness", "Consistency"]
DOC_LEVEL_RUBRIC_NAMES = ["Source Authority", "Timeliness"]


# ---------------- prompt loading (cached) ----------------

_PROMPT_CACHE = None


def _load_prompts():
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        with open(_PROMPT_YAML_PATH, "r", encoding="utf-8") as f:
            _PROMPT_CACHE = yaml.safe_load(f)
    return _PROMPT_CACHE


# ---------------- API client ----------------

def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        # Support both `OpenAI_API_KEY` (used in dr-tulu/agent/.env) and the
        # canonical `OPENAI_API_KEY`.
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set; please configure OpenAI_API_KEY "
                f"in {_DOTENV_PATH}"
            )
        kwargs = {"api_key": api_key, "timeout": HTTP_TIMEOUT}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _OPENAI_CLIENT = OpenAI(**kwargs)
    return _OPENAI_CLIENT


def _call_gpt(prompt_text):
    """Single OpenAI chat completion call. Raises on any error;
    caller handles retries."""
    client = _get_openai_client()
    rsp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt_text}],
    )
    if not rsp.choices:
        raise ValueError("empty 'choices' in OpenAI response")
    value = rsp.choices[0].message.content
    if not value:
        raise ValueError("empty message content in OpenAI response")
    return value


# ---------------- output parsing ----------------

# Match bracketed integers like "[1]", "[12]". Unbracketed ints are NOT
# accepted to avoid false positives from numbers inside any leading text.
_BRACKET_INT_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _parse_selected_ids(predict_str: str):
    """Return list of 0-based indices in their predicted order, or None on
    full failure (no bracketed integer found at all)."""
    if not isinstance(predict_str, str):
        return None
    ids = []
    for m in _BRACKET_INT_RE.finditer(predict_str):
        try:
            ids.append(int(m.group(1)) - 1)
        except ValueError:
            continue
    if not ids:
        return None
    return ids


def _filter_ids(raw_ids, n_initial):
    """Drop out-of-range ids (>= n_initial or < 0); de-dup keeping first."""
    seen = set()
    kept = []
    for x in raw_ids:
        if x < 0 or x >= n_initial:
            continue
        if x in seen:
            continue
        seen.add(x)
        kept.append(x)
    return kept


_INT_RE = re.compile(r"\b(\d{1,2})\b")


def _parse_rubric_scores(text, n_expected):
    """Extract integers in [0,10] from LLM output. The number of valid integers
    found MUST be exactly n_expected; otherwise the output is considered
    untrustworthy and we return None (caller will retry / drop the rubrics).
    Returns list[int] of length n_expected, or None on failure."""
    if not isinstance(text, str):
        return None
    nums = []
    for m in _INT_RE.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 10:
            nums.append(v)
    if len(nums) != n_expected:
        return None
    return nums


# ---------------- rubric extraction ----------------

def _flatten_rubrics(rubric_dict, allowed_names):
    """Flatten a {category: [ {description, weight}, ... ]} dict into a list of
    (category, description, weight). Only categories in `allowed_names` and
    only entries with a non-empty description and a positive weight are kept.

    NOTE: a category may legitimately contain multiple rubrics; we keep them
    all (each contributes independently to the weighted aggregation)."""
    out = []
    if not isinstance(rubric_dict, dict):
        return out
    for name in allowed_names:
        entries = rubric_dict.get(name) or []
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            desc = e.get("description")
            try:
                weight = float(e.get("weight", 0) or 0)
            except (TypeError, ValueError):
                weight = 0.0
            if not desc or weight <= 0:
                continue
            out.append((name, desc, weight))
    return out


# ---------------- set-level scoring ----------------

def _format_rubrics_block(rubric_list):
    """rubric_list: list of (name, description, weight)."""
    lines = ["The following are the scoring rubrics:"]
    for i, (name, desc, _w) in enumerate(rubric_list, 1):
        lines.append(f"{i}. {name}: {desc}")
    return "\n".join(lines)


def _format_doc_set(initial_list, selected_idxs):
    """Render the selected doc subset as '[1] xxx\n\n[2] yyy ...'.
    The numbering inside the prompt is *local* (1..k), not the original ids."""
    parts = []
    for local_i, idx in enumerate(selected_idxs, 1):
        d = initial_list[idx] if idx < len(initial_list) else None
        if isinstance(d, dict):
            content = d.get("content") or ""
        else:
            content = d or ""
        # content = content[:DOC_CONTENT_MAX_CHARS]
        parts.append(f"[{local_i}] {content}")
    return "\n\n".join(parts)


def _build_set_level_prompt(scenario, query, thinking, doc_set, rubric_list):
    prompts = _load_prompts()
    rubrics_block = _format_rubrics_block(rubric_list)
    if scenario == "deepresearch":
        tmpl = prompts["deepresearch"]
        return tmpl.format(
            query=query or "",
            query_intent=thinking or "(not provided)",
            rubrics_block=rubrics_block,
            doc_set=doc_set,
        )
    else:  # rag
        tmpl = prompts["rag"]
        return tmpl.format(
            query=query or "",
            rubrics_block=rubrics_block,
            doc_set=doc_set,
        )


def _score_set_level(scenario, query, thinking, initial_list, selected_idxs,
                     set_level_rubrics):
    """Returns list[float|None] aligned with set_level_rubrics. None means
    'failed' for that rubric (treated as drop in aggregation). On total LLM
    failure, returns a list of all None."""
    if not set_level_rubrics:
        return []
    doc_set = _format_doc_set(initial_list, selected_idxs)
    prompt = _build_set_level_prompt(
        scenario, query, thinking, doc_set, set_level_rubrics
    )
    n = len(set_level_rubrics)
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            value = _call_gpt(prompt)
            parsed = _parse_rubric_scores(value, n)
            if parsed is None:
                raise ValueError(f"parse score failed; raw={value[:200]!r}")
            return [v / 10.0 for v in parsed]
        except Exception as e:
            last_err = e
            logger.warning(
                f"[set-level scoring attempt {attempt}/{MAX_ATTEMPTS}] "
                f"failed: {type(e).__name__}: {e}"
            )
    logger.error(f"set-level scoring exhausted retries; last_err={last_err}")
    return [None] * n


# ---------------- doc-level aggregation (no LLM) ----------------

def _doc_level_mean(initial_list, selected_idxs, rubric_name):
    """Mean of pre-scored values for `rubric_name` over the selected docs,
    ignoring None. Returns None if all selected are None."""
    vals = []
    for idx in selected_idxs:
        if idx >= len(initial_list):
            continue
        d = initial_list[idx]
        if not isinstance(d, dict):
            continue
        v = d.get(rubric_name)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------- main entry ----------------

def compute_score(solution_str, ground_truth=None, extra_info=None, **kwargs):
    """RubricRanker reward.

    Args:
        solution_str: the rollout text (e.g. "[1] [4] [7]").
        ground_truth: unused for this task.
        extra_info: dict with keys
            - initial_list: list of doc dicts (each with content, Source
              Authority, Timeliness). Pre-scored doc-level rubrics live here.
            - rubrics: dict with keys 'set_level_rubrics' and
              'doc_level_rubrics'.
            - scenario: 'deepresearch' or 'rag'.
            - query: the original query string.
            - thinking: optional reasoning (deepresearch only).

    Returns:
        float reward.
    """
    if not isinstance(extra_info, dict):
        return -1.0

    initial_list = extra_info.get("initial_list") or []
    rubrics = extra_info.get("rubrics") or {}
    scenario = extra_info.get("scenario") or "rag"
    query = extra_info.get("query") or ""
    thinking = extra_info.get("thinking") or ""

    # 1) parse rollout output
    raw_ids = _parse_selected_ids(solution_str)
    if raw_ids is None:
        return -1.0
    selected_idxs = _filter_ids(raw_ids, len(initial_list))
    if not selected_idxs:
        return 0.0

    # 2) collect rubrics
    set_level_rubrics = _flatten_rubrics(
        rubrics.get("set_level_rubrics"), SET_LEVEL_RUBRIC_NAMES
    )
    doc_level_rubrics = _flatten_rubrics(
        rubrics.get("doc_level_rubrics"), DOC_LEVEL_RUBRIC_NAMES
    )

    # 3) set-level: ONE LLM call (with retries) covers all set-level rubrics
    set_scores = _score_set_level(
        scenario, query, thinking, initial_list, selected_idxs, set_level_rubrics
    )

    # 4) weighted aggregation
    weighted_sum = 0.0
    total_weight = 0.0

    for (_name, _desc, weight), score in zip(set_level_rubrics, set_scores):
        if score is None:
            continue
        weighted_sum += weight * score
        total_weight += weight

    for (name, _desc, weight) in doc_level_rubrics:
        score = _doc_level_mean(initial_list, selected_idxs, name)
        if score is None:
            continue
        weighted_sum += weight * score
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight
