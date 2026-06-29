"""
SearchWithRerankTool: Combines search → browse → rerank into a single tool.

The agent sees it as a regular search tool (same schema, same output format),
but internally it fetches extra candidates, browses pages for content,
cleans/truncates the text, and reranks before returning top results.
"""

import asyncio
import copy
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool
from .data_types import Document, DocumentToolOutput, ToolInput, ToolOutput
from .tool_parsers import ToolCallParser
from .utils import clean_webpage_text, clean_webpage_text_old

logger = logging.getLogger(__name__)

# Matches the last <think>...</think> block in the assistant's response text.
# The agent prompt (unified_tool_calling_v20250907) instructs the model to
# emit <think>reasoning</think> immediately before <call_tool name="...">...</call_tool>,
# so the trailing <think> block is a good proxy for "search intent".
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Matches the beginning of a tool call, e.g. <call_tool name="search"> ...
# Used to strip the tool-call portion off the tail of the assistant response
# so that only the preceding reasoning/intent text is considered.
_CALL_TOOL_RE = re.compile(r"<call_tool\s+name", re.IGNORECASE)

# Matches a leading <think> or trailing </think> tag (with optional whitespace)
# used as a fallback cleanup when no well-formed <think>...</think> block is found.
_LEADING_THINK_OPEN_RE = re.compile(r"^\s*<think>\s*", re.IGNORECASE)
_TRAILING_THINK_CLOSE_RE = re.compile(r"\s*</think>\s*$", re.IGNORECASE)


def _extract_search_intent_from_response(response_text: str) -> str:
    """Extract the search intent from an assistant response string.

    Strategy:
      1. Strip off the tool-call portion by cutting the text at the first
         occurrence of ``<call_tool name`` (the agent prompt always places
         the tool call after the reasoning). Only the preceding segment is
         considered as candidate intent text.
      2. On the remaining segment, try to extract the content of the last
         ``<think>...</think>`` block (well-formed case).
      3. If no such block is found (e.g. the agent did not emit think tags,
         or only emitted a dangling ``<think>`` without a closing tag), fall
         back to using the entire remaining segment as the intent, after
         stripping any leading ``<think>`` / trailing ``</think>`` tags.

    Returns an empty string if the input is not a usable string.
    """
    if not response_text or not isinstance(response_text, str):
        return ""

    # Step 1: Cut off the tool-call portion (everything from the first
    # <call_tool name ...> onwards). The reasoning/intent is expected to
    # live strictly before the tool call.
    call_match = _CALL_TOOL_RE.search(response_text)
    pre_tool_text = response_text[: call_match.start()] if call_match else response_text

    # Step 2: Prefer a well-formed <think>...</think> block.
    matches = _THINK_BLOCK_RE.findall(pre_tool_text)
    if matches:
        # The last <think> block is the one immediately preceding the
        # triggering <call_tool>.
        return matches[-1].strip()

    # Step 3: Fallback — use the remaining text directly, stripping any
    # dangling <think> / </think> tags at the head/tail.
    fallback = pre_tool_text.strip()
    fallback = _LEADING_THINK_OPEN_RE.sub("", fallback)
    fallback = _TRAILING_THINK_CLOSE_RE.sub("", fallback)
    return fallback.strip()


_QWEN3_8B_TOKENIZER_PATH = "/apdcephfs_zwfy4/share_303945702/wenhanliu/llm/Qwen3-8B"
# Process-level cache: maps tokenizer path -> tokenizer instance, or
# ``False`` if a previous load attempt failed (so we don't retry every call).
_TRUNCATION_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_qwen3_truncation_tokenizer():
    """Return the cached Qwen3-8B tokenizer, or None if loading fails."""
    cached = _TRUNCATION_TOKENIZER_CACHE.get(_QWEN3_8B_TOKENIZER_PATH)
    if cached is not None:
        return cached if cached is not False else None
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
        _TRUNCATION_TOKENIZER_CACHE[_QWEN3_8B_TOKENIZER_PATH] = tok
        return tok
    except Exception as e:
        logger.warning(
            f"[truncate_by_words] failed to load tokenizer "
            f"{_QWEN3_8B_TOKENIZER_PATH}: {e}; falling back to whitespace split"
        )
        _TRUNCATION_TOKENIZER_CACHE[_QWEN3_8B_TOKENIZER_PATH] = False
        return None


def truncate_by_words(text: str, max_words: int) -> str:
    """Truncate ``text`` to at most ``max_words`` Qwen3-8B tokens.

    The parameter name is kept as ``max_words`` for backward-compat with
    the ``doc_max_words`` config knob and existing call sites; the unit is
    semantically *Qwen3-8B tokens*. Whitespace ``split()`` is used as a
    fallback if the tokenizer is unavailable.
    """
    if max_words is None or max_words <= 0 or not text:
        return text
    tok = _get_qwen3_truncation_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= max_words:
                return text
            return tok.decode(ids[:max_words], skip_special_tokens=True) + "..."
        except Exception as e:
            logger.warning(f"[truncate_by_words] tokenizer encode/decode failed: {e}; falling back to whitespace split")
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


class SearchWithRerankTool(BaseTool):
    """
    A search tool with built-in reranking.

    Flow: search → browse pages → clean & truncate → rerank → return top results.

    The agent sees it as a regular search tool (same tool_schema, same output format).
    Internally it:
      1. Searches with more candidates (e.g. 50)
      2. Browses each page to get text content
      3. Builds reranker input: title + snippet + first N words of page text
      4. Reranks using pointwise or listwise reranker
      5. Returns top_n results in original serper format (title + snippet + url)
    """

    def __init__(
        self,
        search_tool: BaseTool,
        browse_tool: BaseTool,
        reranker_tool: BaseTool,
        reranker_top_n: int = 10,
        doc_max_words: int = 300,
        skip_browse: bool = False,
        append_page_content_to_snippet: bool = False,
        tool_parser: Optional[ToolCallParser | str] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(tool_parser=tool_parser, name=name, **kwargs)
        self.search_tool = search_tool
        self.browse_tool = browse_tool
        self.reranker_tool = reranker_tool
        self.reranker_top_n = reranker_top_n
        self.doc_max_words = doc_max_words
        self.skip_browse = skip_browse
        # When True, after reranking the cleaned+truncated page text used
        # for the reranker is appended to each returned document's snippet
        # so the downstream agent gets richer context than just the raw
        # search snippet. When False (default), only the original search
        # snippet is returned.
        self.append_page_content_to_snippet = append_page_content_to_snippet

    async def __call__(
        self, tool_input: Union[str, ToolInput, ToolOutput]
    ) -> DocumentToolOutput:
        call_id = self._generate_call_id()
        start_time = time.time()

        # Step 0: Extract search intent from the assistant response text
        if isinstance(tool_input, str):
            # with open('search_intent.txt', 'a') as f:
            #     f.write(tool_input + '\n\n\n')
            search_intent = _extract_search_intent_from_response(tool_input)
        else:
            search_intent = ""

        # Step 1: Search — get many candidates
        search_output: DocumentToolOutput = await self.search_tool(tool_input)
        if search_output.error or not search_output.documents:
            return search_output  # Pass through errors or empty results

        original_documents = search_output.documents
        logger.info(
            f"SearchWithRerank: got {len(original_documents)} candidates from search"
        )

        # Step 2: Browse — fetch page content (skipped if skip_browse=True)
        browsed_map: Dict[str, Document] = {}
        if not self.skip_browse:
            # print(f"[DEBUG] Starting browse for {len(search_output.documents)} documents")
            browse_output: DocumentToolOutput = await self.browse_tool(search_output)
            # print(f"[DEBUG] Browse done, got {len(browse_output.documents) if browse_output else 0} results") 
            if browse_output and browse_output.documents:
                for doc in browse_output.documents:
                    browsed_map[doc.id] = doc
        else:
            logger.info("SearchWithRerank: skip_browse=True, using title+snippet only")

        # Step 3: Build reranker input — clean and truncate text for each document
        reranker_documents = []
        # Remember, per document id, the cleaned+truncated page text we
        # pass to the reranker. This lets us later (Step 5) optionally
        # append that same text to the returned snippet without having to
        # re-clean/re-truncate the raw browsed text.
        page_text_map: Dict[str, str] = {}
        for orig_doc in original_documents:
            browsed_doc = browsed_map.get(orig_doc.id)
            page_text = ""
            if browsed_doc and browsed_doc.text:
                # Pass the original URL so the cleaner can route PDF scrapes
                # (whose body is often a degenerate "Page-1..Page-N" skeleton)
                # through a PDF-specific filter.
                page_text = clean_webpage_text(
                    browsed_doc.text, url=orig_doc.url or ""
                )
                page_text = truncate_by_words(page_text, self.doc_max_words)
            if page_text:
                page_text_map[orig_doc.id] = page_text

            reranker_text = page_text

            # Create a document copy with the composed text for reranking
            reranker_doc = Document(
                id=orig_doc.id,
                title=orig_doc.title,
                url=orig_doc.url,
                snippet=orig_doc.snippet,
                text=reranker_text,  # Used by reranker's simple_stringify
                score=orig_doc.score,
            )
            reranker_documents.append(reranker_doc)

        # Wrap in DocumentToolOutput for reranker input
        # Merge search_intent into raw_output so rerankers that need it
        # (e.g. rubrics-guided) can pick it up via a stable key without
        # needing a separate plumbing parameter.
        reranker_raw_output: Optional[Dict[str, Any]]
        if isinstance(search_output.raw_output, dict):
            reranker_raw_output = dict(search_output.raw_output)
        else:
            reranker_raw_output = {}
        reranker_raw_output["search_intent"] = search_intent

        reranker_input = DocumentToolOutput(
            tool_name=self.name or "search_with_rerank",
            output="",
            called=True,
            error="",
            timeout=False,
            runtime=0.0,
            call_id=call_id,
            raw_output=reranker_raw_output,
            documents=reranker_documents,
            query=search_output.query,
        )

        # Step 4: Rerank
        rerank_output: DocumentToolOutput = await self.reranker_tool(reranker_input)

        if rerank_output.error or not rerank_output.documents:
            logger.warning(
                f"SearchWithRerank: reranking failed ({rerank_output.error}), "
                f"falling back to original search results"
            )
            # Fallback: return original results (just top_n)
            fallback_docs = original_documents[: self.reranker_top_n]
            return self._build_output(
                fallback_docs, search_output.query, call_id, start_time, search_output
            )

        # Step 5: Map reranked doc ids back to original documents (restore original content)
        original_map = {doc.id: doc for doc in original_documents}
        final_documents = []
        for reranked_doc in rerank_output.documents:
            orig_doc = original_map.get(reranked_doc.id)
            if orig_doc:
                # By default, restore the original snippet only. If the
                # caller opted in to richer snippets, append the
                # cleaned+truncated page text that was shown to the
                # reranker so the downstream agent sees the same context
                # the reranker used to score this document.
                final_snippet = orig_doc.snippet
                if self.append_page_content_to_snippet:
                    page_text = page_text_map.get(orig_doc.id, "")
                    if page_text:
                        if final_snippet:
                            final_snippet = f"{final_snippet}\n{page_text}"
                        else:
                            final_snippet = page_text

                # Restore original content but keep reranked score
                restored = Document(
                    id=orig_doc.id,
                    title=orig_doc.title,
                    url=orig_doc.url,
                    snippet=final_snippet,
                    text=None,  # Don't include browsed text in final output
                    score=reranked_doc.score,
                )
                final_documents.append(restored)

        # Apply top_n limit
        final_documents = final_documents[: self.reranker_top_n]

        logger.info(
            f"SearchWithRerank: returning {len(final_documents)} reranked results"
        )

        return self._build_output(
            final_documents, search_output.query, call_id, start_time, search_output
        )

    def _build_output(
        self,
        documents: List[Document],
        query: Optional[str],
        call_id: str,
        start_time: float,
        search_output: DocumentToolOutput,
    ) -> DocumentToolOutput:
        """Build final DocumentToolOutput in the same format as the original search tool."""
        # Format output string in the same style as SerperSearchTool
        output_parts = []
        for doc in documents:
            output_parts.append(doc.simple_stringify())
        output_str = "\n\n".join(output_parts)

        return DocumentToolOutput(
            tool_name=self.name or "search_with_rerank",
            output=output_str,
            called=True,
            error="",
            timeout=False,
            runtime=time.time() - start_time,
            call_id=call_id,
            raw_output=search_output.raw_output,
            documents=documents,
            query=query,
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        """Use the wrapped search tool's schema so agent sees the same interface."""
        return self.search_tool._generate_tool_schema()
