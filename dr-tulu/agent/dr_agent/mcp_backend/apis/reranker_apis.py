from typing import Any, Dict, List, Optional, Tuple

import cohere
import requests
from pydantic import BaseModel
import json
from ..cache import cached

# Hard-coded token-length limits (kept simple, not user-configurable).
_BGE_TOTAL_MAX_TOKENS = 510            # BGE max_length=512, reserve 2 for [CLS]/[SEP]
_BGE_QUERY_MAX_TOKENS = 96             # cap query first, give remainder to doc
_BGE_DOC_MAX_TOKENS = 400              # hard cap on doc tokens (also bounded by remainder)
_HF_POINTWISE_DOC_MAX_TOKENS = 400     # monoT5/RankT5 max_length=512, leave room for query+template
_LISTWISE_PROMPT_MAX_TOKENS = 3800     # RankVicuna/RankZephyr max_model_len=4096, leave 256 for output
_LISTWISE_DOC_MIN_TOKENS = 16          # don't shrink each doc below this when balancing

_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_tokenizer_cached(model_path: str):
    """Process-level tokenizer cache (used purely for length truncation)."""
    tok = _TOKENIZER_CACHE.get(model_path)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _TOKENIZER_CACHE[model_path] = tok
    return tok


def _truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


class Document(BaseModel):
    """Document model containing the text content."""

    text: str


class RerankResultItem(BaseModel):
    """Reranker result item containing index, relevance score, and document."""

    index: int
    relevance_score: float
    document: Document


class RerankerResult(BaseModel):
    """Reranker result containing a list of RerankResultItem objects."""

    method: str
    model_name: str
    results: List[RerankResultItem]


def vllm_hosted_pointwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """VLLM-hosted pointwise reranker (Cohere rerank API protocol, e.g. BGE)."""
    if top_n == -1:
        top_n = len(documents)

    # Truncate both query and each doc so that query+doc+specials <= 512 (BGE max_length).
    # Critical: if the combined length exceeds 512, vLLM's CUDA kernel will assert and
    # poison the whole CUDA context (the server then needs a restart).
    truncated_query = query
    try:
        tok = _get_tokenizer_cached(model_name)
        # 1) cap query tokens
        truncated_query = _truncate_text_by_tokens(query, tok, _BGE_QUERY_MAX_TOKENS)
        q_len = len(tok.encode(truncated_query, add_special_tokens=False))
        # 2) doc gets the remaining budget (also bounded by hard cap)
        doc_budget = max(64, min(_BGE_DOC_MAX_TOKENS, _BGE_TOTAL_MAX_TOKENS - q_len))
        documents_for_api = [
            _truncate_text_by_tokens(d, tok, doc_budget) for d in documents
        ]
    except Exception as e:
        print(f"[vllm_hosted_pointwise_reranker] tokenizer load failed: {e}; using raw docs")
        documents_for_api = documents

    try:
        client = cohere.ClientV2("sk-fake-key", base_url=api_url)
        rerank_result = client.rerank(model=model_name, query=truncated_query, documents=documents_for_api)
        sorted_results = sorted(
            rerank_result.results, key=lambda x: x.relevance_score, reverse=True
        )
        top_results = [
            RerankResultItem(
                index=result.index,
                relevance_score=result.relevance_score,
                document=Document(text=result.document["text"]),
            )
            for result in sorted_results[:top_n]
        ]
    except Exception as e:
        # Fallback: original order, equal scores
        print(f"[vllm_hosted_pointwise_reranker] failed: {e}; returning original order")
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=1.0 - i * 1e-3,
                document=Document(text=doc),
            )
            for i, doc in enumerate(documents[:top_n])
        ]

    return RerankerResult(
        method="vllm_hosted_pointwise",
        model_name=model_name,
        results=top_results,
    )


# Backward-compat alias
vllm_hosted_reranker = vllm_hosted_pointwise_reranker


# ==================== Listwise Reranker ====================

# NOTE: The OpenAI (data_eval) listwise reranker loads its prompt template
# from ``dr_agent/shared_prompts/openai_listwise_reranking.yaml`` so the
# template is versioned alongside other agent-facing prompts and can
# incorporate the assistant's ``search_intent``.
#
# The vLLM listwise variant keeps its original inline prompt (below, inside
# ``vllm_hosted_listwise_reranker``) for backward compatibility.
_VLLM_LISTWISE_RANKING_PROMPT = """I will provide you with {num_docs} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}

{formatted_docs}

Search Query: {query}

Rank the {num_docs} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [] > ... e.g., [4] > [2] > [5] > [1] > [3]. Only respond with the ranking results, do not say any word or explain."""


class ListwiseRerankerResult(BaseModel):
    """Result from listwise LLM reranker, containing raw ranking output."""

    method: str
    model_name: str
    raw_ranking_output: str  # Raw LLM output, e.g. "[2] > [5] > [1] > ..."
    # Optional fields populated only by the rubrics-guided variant. Kept
    # Optional so existing callers / serialized payloads remain compatible.
    reference_answer: Optional[str] = None
    rubrics_raw: Optional[str] = None


def vllm_hosted_listwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 1024,
) -> ListwiseRerankerResult:
    """
    Listwise reranker using a vLLM-hosted LLM to rank documents.

    The LLM receives all documents at once and outputs a ranked list of doc indices.

    Args:
        query: The query string to rank documents against
        documents: List of document strings to rerank
        top_n: Number of top documents to return (passed for client-side use)
        model_name: Name of the LLM model
        api_url: Base URL for the vLLM API (e.g. "http://localhost:30003/v1")
        max_tokens: Max tokens for LLM generation

    Returns:
        ListwiseRerankerResult with raw LLM ranking output
    """
    # Format documents with numbered identifiers
    formatted_docs = "\n\n".join(
        f"[{i + 1}] {doc}" for i, doc in enumerate(documents)
    )

    prompt = _VLLM_LISTWISE_RANKING_PROMPT.format(
        num_docs=len(documents),
        query=query,
        formatted_docs=formatted_docs,
    )
    # Disable thinking mode for models like Qwen3 to get clean ranking output
    # Use a long timeout since listwise ranking prompts can be very large
    response = requests.post(
        f"{api_url}/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"].strip()

    return ListwiseRerankerResult(
        method="vllm_hosted_listwise",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )




# ==================== Rubrics-Guided Listwise Reranker ====================

# Progress: load prompt templates from data_process/prompts lazily and
# cache them in module-level dicts. Also cache (query, search_intent) ->
# (reference_answer, rubrics_raw) within a single process, since the same
# agent can invoke multiple search tools that share query/intent and
# re-generating those every time is expensive.

import hashlib
import os
import threading
import yaml

# data_process/prompts path, resolved relative to this file's location:
#   .../dr-tulu/agent/dr_agent/mcp_backend/apis/reranker_apis.py
#   .../data_process/prompts/*.yaml
_THIS_FILE = os.path.abspath(__file__)
_PROMPT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(_THIS_FILE),
        "..", "..", "..", "..", "..",
        "data_process", "prompts",
    )
)

# dr_agent/shared_prompts path, resolved relative to this file's location:
#   .../dr-tulu/agent/dr_agent/mcp_backend/apis/reranker_apis.py
#   .../dr-tulu/agent/dr_agent/shared_prompts/*.yaml
_SHARED_PROMPT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(_THIS_FILE),
        "..", "..",
        "shared_prompts",
    )
)

_PROMPT_CACHE: Dict[str, str] = {}
_PROMPT_CACHE_LOCK = threading.Lock()


def _load_prompt_yaml(
    filename: str,
    key: str,
    base_dir: Optional[str] = None,
) -> str:
    """Load a named prompt template from <base_dir>/<filename>.

    Args:
        filename: YAML filename, e.g. ``"rubrics_generation.yaml"``.
        key: Top-level key inside the YAML whose value is the template.
        base_dir: Directory containing the YAML. When ``None``, defaults
            to ``_PROMPT_DIR`` (i.e. ``data_process/prompts``). Use
            ``_SHARED_PROMPT_DIR`` to load from
            ``dr_agent/shared_prompts``.

    Results are memoized by (base_dir, filename, key).
    """
    resolved_base = base_dir if base_dir is not None else _PROMPT_DIR
    cache_key = f"{resolved_base}::{filename}::{key}"
    with _PROMPT_CACHE_LOCK:
        cached_val = _PROMPT_CACHE.get(cache_key)
        if cached_val is not None:
            return cached_val

    path = os.path.join(resolved_base, filename)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if key not in cfg:
        raise KeyError(
            f"Key {key!r} not found in {path}; available keys: {list(cfg.keys())}"
        )
    tpl = cfg[key]

    with _PROMPT_CACHE_LOCK:
        _PROMPT_CACHE[cache_key] = tpl
    return tpl


# (query, search_intent, model_name) -> (reference_answer, rubrics_raw)
_RUBRICS_CACHE: Dict[str, Dict[str, str]] = {}
_RUBRICS_CACHE_LOCK = threading.Lock()


def _rubrics_cache_key(query: str, search_intent: str, model_name: str) -> str:
    h = hashlib.sha256()
    h.update(query.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(search_intent.encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    h.update(model_name.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _generate_reference_answer(
    query: str,
    search_intent: str,
    model_name: str,
    host: Optional[str],
    port: Optional[int],
    max_retries: int,
    timeout: int,
) -> str:
    """Step A: generate a reference answer for (query, search_intent) using web_search."""
    from .openai_apis import chat as openai_chat

    long_tpl = _load_prompt_yaml("answer_generation.yaml", "long_form")
    prompt = long_tpl.format(query=query, query_intent=search_intent)

    answer = openai_chat(
        prompt_text=prompt,
        model_marker=model_name,
        web_search=True,  # Step A always uses web search.
        max_retries=max_retries,
        host=host,
        port=port,
        timeout=timeout,
    )
    if not isinstance(answer, str):
        answer = str(answer) if answer else ""
    return answer.strip()


def _generate_rubrics(
    query: str,
    search_intent: str,
    reference_answer: str,
    model_name: str,
    host: Optional[str],
    port: Optional[int],
    max_retries: int,
    timeout: int,
) -> str:
    """Step B: generate rubrics JSON string from (query, search_intent, reference_answer)."""
    from .openai_apis import chat as openai_chat

    rubrics_tpl = _load_prompt_yaml("rubrics_generation.yaml", "rubrics_generation")
    prompt = rubrics_tpl.format(
        query=query,
        query_intent=search_intent,
        reference_answer=reference_answer,
    )

    rubrics_raw = openai_chat(
        prompt_text=prompt,
        model_marker=model_name,
        web_search=False,  # Pure reasoning; no web search needed.
        max_retries=max_retries,
        host=host,
        port=port,
        timeout=timeout,
    )
    if not isinstance(rubrics_raw, str):
        rubrics_raw = str(rubrics_raw) if rubrics_raw else ""
    return rubrics_raw.strip()


def _get_or_compute_reference_and_rubrics(
    query: str,
    search_intent: str,
    model_name: str,
    host: Optional[str],
    port: Optional[int],
    max_retries: int,
    timeout: int,
) -> Dict[str, str]:
    """Return cached (reference_answer, rubrics_raw) for (query, search_intent).

    Step A and Step B are the expensive parts of the rubrics-guided
    reranker (each is a data_eval call; Step A also uses web search). We
    memoize them keyed on (query, search_intent, model_name) so that
    multiple search tool invocations with the same query/intent within a
    single MCP server process reuse the computation.
    """
    cache_key = _rubrics_cache_key(query, search_intent, model_name)
    with _RUBRICS_CACHE_LOCK:
        hit = _RUBRICS_CACHE.get(cache_key)
        if hit is not None:
            return hit

    reference_answer = _generate_reference_answer(
        query=query,
        search_intent=search_intent,
        model_name=model_name,
        host=host,
        port=port,
        max_retries=max_retries,
        timeout=timeout,
    )
    rubrics_raw = _generate_rubrics(
        query=query,
        search_intent=search_intent,
        reference_answer=reference_answer,
        model_name=model_name,
        host=host,
        port=port,
        max_retries=max_retries,
        timeout=timeout,
    )
    entry = {
        "reference_answer": reference_answer,
        "rubrics_raw": rubrics_raw,
    }
    with _RUBRICS_CACHE_LOCK:
        _RUBRICS_CACHE[cache_key] = entry
    return entry



# ==================== HF Pointwise Reranker (monoT5 / RankT5) ====================

# Process-level cache so the model is loaded once per MCP server.
_HF_RERANKER_CACHE: Dict[str, Any] = {}
_HF_RERANKER_LOCK = threading.Lock()


# Tokenizer used to length-bound documents passed to rerankers. We use
# Qwen3-8B's tokenizer regardless of the actual reranker model because we
# want a *consistent*, language-aware token count across all reranker
# variants (BGE / monoT5 / RankT5 / RankVicuna / RankQwen / RubricRanker /
# data_eval listwise). Plain ``str.split()`` was previously used, but it
# silently no-ops on CJK text (no whitespace -> 1 "word"), so configured
# ``doc_max_words`` had no effect on Chinese pages.
_QWEN3_8B_TOKENIZER_PATH = "/apdcephfs_zwfy4/share_303945702/wenhanliu/llm/Qwen3-8B"


def _get_qwen3_truncation_tokenizer():
    """Return the cached Qwen3-8B tokenizer, or None if loading fails."""
    tok = _TOKENIZER_CACHE.get(_QWEN3_8B_TOKENIZER_PATH)
    if tok is not None:
        return tok if tok is not False else None
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(
                _QWEN3_8B_TOKENIZER_PATH, use_fast=True
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                _QWEN3_8B_TOKENIZER_PATH, use_fast=False
            )
        _TOKENIZER_CACHE[_QWEN3_8B_TOKENIZER_PATH] = tok
        return tok
    except Exception as e:
        print(
            f"[_get_qwen3_truncation_tokenizer] failed to load "
            f"{_QWEN3_8B_TOKENIZER_PATH}: {e}; falling back to whitespace split"
        )
        # Sentinel so we don't retry on every call.
        _TOKENIZER_CACHE[_QWEN3_8B_TOKENIZER_PATH] = False
        return None


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate ``text`` to at most ``max_words`` Qwen3-8B tokens.

    The parameter is still named ``max_words`` for backward compatibility
    with the existing call sites and the ``doc_max_words`` config knob;
    the unit semantically is *Qwen3-8B tokens* (which is a much better
    proxy for prompt length than whitespace-delimited words, and which
    correctly bounds CJK text).

    Falls back to whitespace truncation if the tokenizer cannot be loaded
    or fails on this particular input.
    """
    if max_words is None or max_words <= 0 or not text:
        return text
    tok = _get_qwen3_truncation_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= max_words:
                return text
            return tok.decode(ids[:max_words], skip_special_tokens=True)
        except Exception as e:
            print(f"[_truncate_words] tokenizer encode/decode failed: {e}; falling back to whitespace split")
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _get_or_load_hf_reranker(model_path: str, model_kind: str, device: str):
    """Load (model, tokenizer) once per process, keyed by model_path."""
    with _HF_RERANKER_LOCK:
        cached = _HF_RERANKER_CACHE.get(model_path)
        if cached is not None:
            return cached

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            T5ForConditionalGeneration,
            T5Tokenizer,
        )

        kind = (model_kind or "").lower()
        if kind == "monot5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        elif kind == "rankt5":
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
            # rankt5 ckpt typically lacks a usable tokenizer; reuse monoT5's spiece if available,
            # otherwise fall back to the same path.
            try:
                tokenizer = T5Tokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        else:
            raise ValueError(f"Unsupported HF reranker model_kind: {model_kind}")

        _HF_RERANKER_CACHE[model_path] = (model, tokenizer, kind)
        return _HF_RERANKER_CACHE[model_path]


def hf_pointwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_path: str,
    model_kind: str,
    device: str = "cuda:0",
    doc_max_words: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
) -> RerankerResult:
    """HF in-process pointwise reranker for monoT5 / RankT5."""
    if top_n == -1 or top_n is None:
        top_n = len(documents)

    try:
        import torch

        model, tokenizer, kind = _get_or_load_hf_reranker(model_path, model_kind, device)

        q = _truncate_text_by_tokens(query, tokenizer, _BGE_QUERY_MAX_TOKENS)
        docs_trunc = [
            _truncate_text_by_tokens(
                _truncate_words(d, doc_max_words), tokenizer, _HF_POINTWISE_DOC_MAX_TOKENS
            )
            for d in documents
        ]

        if kind == "monot5":
            patterns = [f"Query: {q} Document: {d} Relevant:" for d in docs_trunc]
        else:  # rankt5
            patterns = [f"Query: {q} Document: {d}" for d in docs_trunc]

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(patterns), batch_size):
                batch = patterns[i:i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True,
                ).to(device)
                outputs = model.generate(
                    **inputs,
                    max_length=2,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                first_step_logits = outputs.scores[0]  # (B, V)
                if kind == "monot5":
                    # token ids for "true" / "false"
                    s = torch.nn.functional.log_softmax(
                        first_step_logits[:, [1176, 6136]], dim=1
                    )[:, 0]
                else:  # rankt5: <extra_id_10> -> 32089
                    s = first_step_logits[:, 32089]
                scores.extend(s.float().cpu().tolist())

        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_n]
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=float(scores[i]),
                document=Document(text=documents[i]),
            )
            for i in order
        ]
        method = f"hf_pointwise_{kind}"
    except Exception as e:
        print(f"[hf_pointwise_reranker] failed: {e}; returning original order")
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=1.0 - i * 1e-3,
                document=Document(text=doc),
            )
            for i, doc in enumerate(documents[:top_n])
        ]
        method = f"hf_pointwise_{model_kind}_fallback"

    return RerankerResult(
        method=method,
        model_name=model_path,
        results=top_results,
    )


# ==================== Listwise Sliding-Window Reranker (RankVicuna / RankZephyr) ====================

_LISTWISE_PROMPT_CACHE: Dict[str, Dict[str, str]] = {}


def _load_listwise_prompt() -> Dict[str, str]:
    if "tpl" in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE["tpl"]
    path = os.path.join(_SHARED_PROMPT_DIR, "reranker_prompts", "listwise_reranker.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE["tpl"] = cfg
    return cfg


def _build_vicuna_prompt(system_message: str, user_content: str) -> str:
    # fastchat vicuna_v1.1 template (matches castorini/rank_vicuna_7b_v1)
    sys = system_message or ""
    return (
        f"{sys}\n\n"
        f"USER: {user_content}\n"
        f"ASSISTANT:"
    )


def _build_zephyr_prompt(system_message: str, user_content: str) -> str:
    # zephyr/mistral chat template
    sys = system_message or ""
    return (
        f"<|system|>\n{sys}</s>\n"
        f"<|user|>\n{user_content}</s>\n"
        f"<|assistant|>\n"
    )


def _build_listwise_user_content(query: str, documents: List[str], doc_max_words: int) -> Tuple[str, int]:
    cfg = _load_listwise_prompt()
    prefix = cfg.get("prefix", "").format(num=len(documents), query=query)
    body_tpl = cfg.get("body", "[{rank}] {candidate}\n")
    suffix = cfg.get("suffix", "").format(num=len(documents), query=query)

    parts = [prefix]
    for i, d in enumerate(documents):
        d_trunc = _truncate_words(d, doc_max_words)
        parts.append(body_tpl.format(rank=i + 1, candidate=d_trunc))
    parts.append(suffix)
    return "".join(parts), len(documents)


def _build_listwise_prompt_with_token_limit(
    query: str,
    window_docs: List[str],
    doc_max_words: int,
    system_message: str,
    model_kind: str,
    tokenizer,
) -> str:
    """Build a sliding-window prompt and shrink each doc until it fits in the model context.

    On any failure (tokenizer issues, cannot shrink enough), returns the un-shrunk prompt.
    """
    def _assemble(docs_for_window: List[str]) -> str:
        user_content, _ = _build_listwise_user_content(query, docs_for_window, 0)
        if (model_kind or "").lower() == "zephyr":
            return _build_zephyr_prompt(system_message, user_content)
        return _build_vicuna_prompt(system_message, user_content)

    # Word-level pre-truncate.
    docs = [_truncate_words(d, doc_max_words) for d in window_docs]

    if tokenizer is None:
        return _assemble(docs)

    try:
        prompt = _assemble(docs)
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens <= _LISTWISE_PROMPT_MAX_TOKENS:
            return prompt

        # Iteratively halve each doc's per-doc token budget until prompt fits.
        # Start budget = average current doc tokens.
        doc_token_lists = [tokenizer.encode(d, add_special_tokens=False) for d in docs]
        budget = max(
            _LISTWISE_DOC_MIN_TOKENS,
            int(sum(len(t) for t in doc_token_lists) / max(1, len(docs))),
        )
        for _ in range(8):
            budget = max(_LISTWISE_DOC_MIN_TOKENS, budget // 2 if budget > _LISTWISE_DOC_MIN_TOKENS else budget)
            shrunk = [
                tokenizer.decode(t[:budget], skip_special_tokens=True) for t in doc_token_lists
            ]
            prompt = _assemble(shrunk)
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if n_tokens <= _LISTWISE_PROMPT_MAX_TOKENS:
                return prompt
            if budget <= _LISTWISE_DOC_MIN_TOKENS:
                break
        return prompt
    except Exception as e:
        print(f"[_build_listwise_prompt_with_token_limit] tokenizer error: {e}; using raw prompt")
        return _assemble(docs)


def _parse_permutation(raw: str) -> List[int]:
    """Parse "[2] > [5] > [1] > ..." into 0-based indices."""
    if not raw:
        return []
    cleaned = "".join(c if c.isdigit() else " " for c in raw)
    tokens = cleaned.split()
    out: List[int] = []
    seen = set()
    for t in tokens:
        try:
            v = int(t) - 1
        except ValueError:
            continue
        if v >= 0 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _apply_permutation_window(order: List[int], perm: List[int], start: int, end: int) -> List[int]:
    """Reorder order[start:end] according to perm (0-based, relative to window)."""
    window = order[start:end]
    n = len(window)
    perm = [p for p in perm if 0 <= p < n]
    seen = set(perm)
    perm = perm + [i for i in range(n) if i not in seen]
    new_window = [window[p] for p in perm]
    return order[:start] + new_window + order[end:]


def vllm_listwise_sliding_window_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    model_kind: str = "vicuna",
    window_size: int = 20,
    step: int = 10,
    max_tokens: int = 200,
    doc_max_words: int = 0,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Sliding-window listwise reranker for RankVicuna / RankZephyr.

    Calls vLLM's /v1/completions (raw text) with a model-specific chat template
    assembled in-process. Slides from the tail toward the head, step by step.
    """
    n = len(documents)
    if n == 0:
        return ListwiseRerankerResult(
            method="vllm_listwise_sliding_window",
            model_name=model_name,
            raw_ranking_output="",
        )

    cfg = _load_listwise_prompt()
    system_message = cfg.get("system_message", "")

    base_url = api_url.rstrip("/")
    completions_url = f"{base_url}/completions"

    order = list(range(n))
    last_raw = ""

    # Tokenizer used only for length-bounding the prompt.
    try:
        tokenizer_for_len = _get_tokenizer_cached(model_name)
    except Exception as e:
        print(f"[vllm_listwise_sliding_window_reranker] tokenizer load failed: {e}; skipping length cap")
        tokenizer_for_len = None

    try:
        windows_end = n
        windows_start = max(0, n - window_size)
        # iterate backwards
        while windows_end > 0:
            ws = max(0, windows_start)
            we = windows_end
            window_docs = [documents[order[i]] for i in range(ws, we)]
            prompt = _build_listwise_prompt_with_token_limit(
                query=query,
                window_docs=window_docs,
                doc_max_words=doc_max_words,
                system_message=system_message,
                model_kind=model_kind,
                tokenizer=tokenizer_for_len,
            )

            resp = requests.post(
                completions_url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"]
            last_raw = text
            perm = _parse_permutation(text)
            order = _apply_permutation_window(order, perm, ws, we)

            if ws == 0:
                break
            windows_end -= step
            windows_start -= step

        raw_output = " > ".join(f"[{i + 1}]" for i in order)
    except Exception as e:
        print(f"[vllm_listwise_sliding_window_reranker] failed: {e}; returning original order")
        raw_output = ""

    return ListwiseRerankerResult(
        method=f"vllm_listwise_sliding_window_{model_kind}",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Setwise Rank4Gen Reranker ====================

def _load_rank4gen_prompt() -> Dict[str, str]:
    cache_key = "rank4gen"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = os.path.join(_SHARED_PROMPT_DIR, "reranker_prompts", "rank4gen.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def vllm_setwise_rank4gen_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    downstream_model: str = "default",
    downstream_desc: str = "default",
    max_tokens: int = 4096,
    doc_max_words: int = 0,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Setwise reranker for Rank4Gen. Uses chat completions; English prompt only."""
    cfg = _load_rank4gen_prompt()
    system_message = cfg.get("system_message", "").format(
        downstream_model=downstream_model,
        description=downstream_desc,
    )
    user_tpl = cfg.get("user_message", "")
    mode_tag = cfg.get("mode_tag", "/index")

    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    context = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))
    user_content = user_tpl.format(num=len(documents), question=query, context=context) + "\n" + mode_tag

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_rank4gen_reranker] failed: {e}; returning empty output")
        raw_output = ""
        
    return ListwiseRerankerResult(
        method="vllm_setwise_rank4gen",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Setwise SETR Reranker ====================

def _load_setr_prompt() -> str:
    cache_key = "setr"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]["prompt"]
    path = os.path.join(_SHARED_PROMPT_DIR, "reranker_prompts", "setr.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg["prompt"]


def vllm_setwise_setr_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 4096,
    doc_max_words: int = 0,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Setwise reranker for SETR. English prompt only."""
    tpl = _load_setr_prompt()
    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))
    prompt = tpl.format(num=len(documents), question=query, context=context)

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_setr_reranker] failed: {e}; returning empty output")
        raw_output = ""


    return ListwiseRerankerResult(
        method="vllm_setwise_setr",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )

# ==================== Setwise RubricRanker Reranker ====================

def _load_rubricranker_prompt(variant: str) -> Dict[str, str]:
    """Load (and cache) one variant ('deepresearch' or 'rag') of the
    RubricRanker prompt template from rubricranker.yaml.
    """
    cache_key = f"rubricranker::{variant}"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = os.path.join(_SHARED_PROMPT_DIR, "reranker_prompts", "rubricranker.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if variant not in cfg:
        raise ValueError(
            f"rubricranker.yaml missing variant '{variant}'. "
            f"Available: {list(cfg.keys())}"
        )
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg[variant]
    return cfg[variant]

def vllm_setwise_rubricranker_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    search_intent: str = "",
    prompt_variant: str = "deepresearch",
    max_tokens: int = 1024,
    doc_max_words: int = 0,
    timeout: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> ListwiseRerankerResult:
    """Setwise reranker for the student RubricRanker model.

    Two prompt variants:
      - "deepresearch": uses {num, question, query_intent, context}.
      - "rag":          uses {num, question, context}.

    The student model has the 5 rubrics baked into its system message and
    outputs a selection like "[3] [7] [9] ...". Sampling defaults follow
    Qwen-style chat: temperature=0.7, top_p=0.8, top_k=20, min_p=0.
    """
    cfg = _load_rubricranker_prompt(prompt_variant)
    system_message = cfg.get("system_message", "")
    user_tpl = cfg.get("user_message", "")

    docs_trunc = [_truncate_words(d, 300) for d in documents]
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))

    fmt_kwargs: Dict[str, Any] = {
        "num": len(documents),
        "question": query,
        "context": context,
    }
    if prompt_variant == "deepresearch":
        fmt_kwargs["query_intent"] = search_intent or ""
    user_content = user_tpl.format(**fmt_kwargs)

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "extra_body": {
                    "top_k": top_k,
                    "min_p": min_p,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_rubricranker_reranker] failed: {e}; returning empty output")
        raw_output = ""
        
    return ListwiseRerankerResult(
        method="vllm_setwise_rubricranker",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )

# ==================== Listwise Single-Pass Reranker (RankQwen-style) ====================

def _load_listwise_singlepass_prompt() -> Dict[str, str]:
    """Load (and cache) the RankZephyr-style listwise prompt from
    ``shared_prompts/reranker_prompts/listwise_reranker.yaml``.
    Returns dict with keys: system_message, prefix, body, suffix.
    """
    cache_key = "listwise_singlepass::default"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = os.path.join(
        _SHARED_PROMPT_DIR, "reranker_prompts", "listwise_reranker.yaml"
    )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def vllm_listwise_singlepass_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 1024,
    doc_max_words: int = 0,
    timeout: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> ListwiseRerankerResult:
    """Single-pass listwise reranker (RankQwen-style).

    Uses the RankZephyr training prompt (``listwise_reranker.yaml``) but
    sends it as system+user via chat-completions (no sliding window).
    Output format: ``"[2] > [5] > [1] > ..."``; client-side parses and
    truncates to top_n.
    """
    cfg = _load_listwise_singlepass_prompt()
    system_message = cfg.get("system_message", "")
    prefix_tpl = cfg.get("prefix", "")
    body_tpl = cfg.get("body", "")
    suffix_tpl = cfg.get("suffix", "")

    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    num = len(docs_trunc)
    parts = [prefix_tpl.format(num=num, query=query)]
    for i, d in enumerate(docs_trunc):
        parts.append(body_tpl.format(rank=i + 1, candidate=d))
    parts.append(suffix_tpl.format(num=num, query=query))
    user_content = "".join(parts)

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "extra_body": {
                    "top_k": top_k,
                    "min_p": min_p,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_listwise_singlepass_reranker] failed: {e}; returning empty output")
        raw_output = ""

    return ListwiseRerankerResult(
        method="vllm_listwise_singlepass",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )
