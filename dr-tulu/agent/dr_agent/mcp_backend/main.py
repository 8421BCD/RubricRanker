import argparse
import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Annotated, List, Optional

import aiohttp
from transformers import AutoTokenizer
import dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from .apis.data_model import Crawl4aiApiResult
from .apis.massive_serve_apis import parse_massive_serve_results, search_massive_serve
from .apis.pubmed_apis import search_pubmed
from .apis.reranker_apis import ListwiseRerankerResult, RerankerResult
from .apis.semantic_scholar_apis import (
    SemanticScholarSearchQueryParams,
    SemanticScholarSnippetSearchQueryParams,
    search_semantic_scholar_keywords,
    search_semantic_scholar_snippets,
)
from .apis.serper_apis import (
    ScholarResponse,
    SearchResponse,
    WebpageContentResponse,
    fetch_webpage_content,
    search_serper,
    search_serper_scholar,
)
from .apis.jina_apis import JinaWebpageResponse, fetch_webpage_content_jina
from .cache import set_cache_enabled


def _is_pdf_url(url: str) -> bool:
    """Return True if the URL points to a PDF (by path suffix, ignoring query/fragment)."""
    if not url:
        return False
    try:
        from urllib.parse import urlparse

        path = urlparse(url).path or ""
    except Exception:
        path = url
    return path.lower().endswith(".pdf")


def _sanitize_metadata(meta) -> dict:
    """Coerce arbitrary scraper-provided metadata into the schema expected by
    ``WebpageContentResponse`` / ``JinaMetadata`` (``Dict[str, str|int|bool]``).

    Real-world scrapers (Serper /scrape, Jina) sometimes return metadata values
    that are ``None``, nested ``dict``/``list`` (when a page has duplicate or
    structured ``<meta>`` tags, e.g. ``{"name": "False"}``), which makes
    FastMCP's output validation reject the entire response. We keep ``str``,
    ``int``, ``bool`` as-is, drop ``None``, and stringify everything else.
    """
    if not isinstance(meta, dict):
        return {}
    cleaned: dict = {}
    for key, value in meta.items():
        if value is None:
            continue
        try:
            key_str = str(key)
        except Exception:
            continue
        # bool is a subclass of int -- check it first so True/False stay as bool.
        if isinstance(value, (bool, int, str)):
            cleaned[key_str] = value
        else:
            try:
                cleaned[key_str] = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                cleaned[key_str] = str(value)
    return cleaned


def _jina_to_serper_response(
    jina_resp: dict, url: str
) -> dict:
    """Convert a JinaWebpageResponse into the serper WebpageContentResponse schema
    so downstream clients (which expect ``text`` / ``markdown`` / ``metadata``)
    keep working unchanged.
    """
    content = jina_resp.get("content") or ""
    metadata = _sanitize_metadata(jina_resp.get("metadata"))
    title = jina_resp.get("title") or ""
    if title and "title" not in metadata:
        metadata["title"] = title
    description = jina_resp.get("description") or ""
    if description and "description" not in metadata:
        metadata["description"] = description
    published_time = jina_resp.get("publishedTime") or ""
    if published_time and "publishedTime" not in metadata:
        metadata["publishedTime"] = published_time
    return {
        "text": content,
        "markdown": content,
        "metadata": metadata,
        "url": jina_resp.get("url") or url,
        "success": True,
    }
from .local.crawl4ai_fetcher import Crawl4AiResult
from .local.search import SearcherType
from .local.search.base import LocalSearchResponse

# Global instance for local search
local_searcher = None
snippet_tokenizer = None
snippet_max_tokens = 0

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

mcp = FastMCP(
    "RL-RAG MCP",
    include_tags=os.environ.get("MCP_INCLUDE_TAGS", "search,browse,rerank").split(","),
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Check if the MCP server is running.
    curl http://127.0.0.1:8000/health
    """
    return PlainTextResponse("OK")


@mcp.tool(tags={"search", "necessary"})
def semantic_scholar_search(
    query: Annotated[str, "Search query string"],
    year: Annotated[
        Optional[str], "Year range filter (e.g., '2015-2020', '2015-', '-2015')"
    ] = None,
    min_citation_count: Annotated[Optional[int], "Minimum number of citations"] = None,
    sort: Annotated[
        Optional[str], "Sort order (e.g., 'citationCount:asc', 'publicationDate:desc')"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Maximum number of results to return (max: 100)"] = 25,
) -> dict:
    """
    Search for academic papers using Semantic Scholar API.

    Returns:
        Dictionary containing search results
    """
    query_params = SemanticScholarSearchQueryParams(
        query=query,
        year=year,
        minCitationCount=min_citation_count,
        sort=sort,
        venue=venue,
    )

    results = search_semantic_scholar_keywords(
        query_params=query_params,
        limit=min(limit, 100),  # Ensure limit doesn't exceed API maximum
    )

    return results


@mcp.tool(tags={"search"})
def semantic_scholar_snippet_search(
    query: Annotated[str, "Search query string to find within paper content"],
    year: Annotated[
        Optional[str],
        "Publication year filter - single number (e.g., '2024') or range (e.g., '2022-2025', '2020-', '-2023')",
    ] = None,
    paper_ids: Annotated[
        Optional[str], "Comma-separated list of specific paper IDs to search within"
    ] = None,
    venue: Annotated[Optional[str], "Venue filter (e.g., 'ACL', 'EMNLP')"] = None,
    limit: Annotated[int, "Number of snippets to retrieve"] = 10,
) -> dict:
    """
    Focused snippet retrieval from scientific papers using Semantic Scholar API.

    Purpose: Search for specific text snippets within academic papers to find relevant passages, quotes,
    or mentions from scientific literature. Returns focused snippets from existing papers rather than
    full paper metadata.

    Returns:
        Dictionary containing snippets from existing papers with text passages and their source papers.
        Each snippet includes the relevant text passage and metadata about the source paper.

    Example:
        Search for LLM evaluation snippets published between 2021-2025 in CS/Medicine:
        query="large language model retrieval evaluation", year="2021-2025", limit=8
    """
    # Convert comma-separated string to list if provided
    paper_ids_list = None
    if paper_ids:
        paper_ids_list = [pid.strip() for pid in paper_ids.split(",")]

    query_params = SemanticScholarSnippetSearchQueryParams(
        query=query,
        year=year,
        paperIds=paper_ids_list,
        venue=venue,
    )

    results = search_semantic_scholar_snippets(
        query_params=query_params,
        limit=limit,
    )

    return results


@mcp.tool(tags={"search"})
def pubmed_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """
    Search for medical and scientific papers using PubMed API.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default: 10)
        offset: Starting position for pagination (default: 0)

    Returns:
        Dictionary containing search results with the following fields:
        - total: Total number of results
        - offset: Current offset
        - next: Next offset for pagination
        - data: List of paper details including:
            - paperId: PubMed ID
            - title: Paper title
            - authors: List of authors
            - abstract: Paper abstract
            - year: Publication year
            - venue: Journal name
            - url: Link to PubMed page
            - citationCount: Number of citations (if available from Semantic Scholar)
    """
    results = search_pubmed(
        keywords=query,
        limit=limit,
        offset=offset,
    )

    return results


@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_pointwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """VLLM-hosted pointwise reranker (Cohere rerank API protocol, e.g. BGE)."""
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_pointwise_reranker

    return vllm_hosted_pointwise_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
    )


# Backward-compat alias for the original tool name.
@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
) -> RerankerResult:
    """Deprecated alias of vllm_hosted_pointwise_reranker."""
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_pointwise_reranker

    return vllm_hosted_pointwise_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
    )


@mcp.tool(tags={"necessary", "rerank"})
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
    """In-process HF pointwise reranker for monoT5 / RankT5."""
    from dr_agent.mcp_backend.apis.reranker_apis import hf_pointwise_reranker

    return hf_pointwise_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_path=model_path,
        model_kind=model_kind,
        device=device,
        doc_max_words=doc_max_words,
        batch_size=batch_size,
        max_length=max_length,
    )


@mcp.tool(tags={"necessary", "rerank"})
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
    """Sliding-window listwise reranker for RankVicuna / RankZephyr."""
    from dr_agent.mcp_backend.apis.reranker_apis import (
        vllm_listwise_sliding_window_reranker,
    )

    return vllm_listwise_sliding_window_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        model_kind=model_kind,
        window_size=window_size,
        step=step,
        max_tokens=max_tokens,
        doc_max_words=doc_max_words,
        timeout=timeout,
    )


@mcp.tool(tags={"necessary", "rerank"})
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
    """Setwise reranker for Rank4Gen (English prompt)."""
    from dr_agent.mcp_backend.apis.reranker_apis import (
        vllm_setwise_rank4gen_reranker,
    )

    return vllm_setwise_rank4gen_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        downstream_model=downstream_model,
        downstream_desc=downstream_desc,
        max_tokens=max_tokens,
        doc_max_words=doc_max_words,
        timeout=timeout,
    )


@mcp.tool(tags={"necessary", "rerank"})
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
    """Setwise reranker for SETR (English prompt)."""
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_setwise_setr_reranker

    return vllm_setwise_setr_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        max_tokens=max_tokens,
        doc_max_words=doc_max_words,
        timeout=timeout,
    )


@mcp.tool(tags={"necessary", "rerank"})
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

    Supports two prompt variants:
      - "deepresearch": uses query + agent's search_intent (English).
      - "rag":          plain user question, no search_intent.
    """
    from dr_agent.mcp_backend.apis.reranker_apis import (
        vllm_setwise_rubricranker_reranker,
    )

    return vllm_setwise_rubricranker_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        search_intent=search_intent,
        prompt_variant=prompt_variant,
        max_tokens=max_tokens,
        doc_max_words=doc_max_words,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )


@mcp.tool(tags={"necessary", "rerank"})
def vllm_hosted_listwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 1024,
) -> ListwiseRerankerResult:
    """
    Rerank documents using a listwise LLM reranker hosted on vLLM.

    The LLM receives all documents and outputs a ranked list like "[2] > [5] > [1] > ...".

    Args:
        query: Search query string
        documents: List of document texts to rank
        top_n: Number of top documents to return
        model_name: Name of the LLM reranker model
        api_url: Base URL for the vLLM API (e.g. "http://localhost:30003/v1")
        max_tokens: Max tokens for LLM generation

    Returns:
        ListwiseRerankerResult with raw LLM ranking output
    """
    from dr_agent.mcp_backend.apis.reranker_apis import vllm_hosted_listwise_reranker

    results = vllm_hosted_listwise_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        max_tokens=max_tokens,
    )

    return results


@mcp.tool(tags={"necessary", "rerank"})
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

    Uses the RankZephyr training prompt (system + user via chat-completions),
    no sliding window. Output is "[2] > [5] > [1] > ..."; client-side
    truncates to top_n.
    """
    from dr_agent.mcp_backend.apis.reranker_apis import (
        vllm_listwise_singlepass_reranker,
    )

    return vllm_listwise_singlepass_reranker(
        query=query,
        documents=documents,
        top_n=top_n,
        model_name=model_name,
        api_url=api_url,
        max_tokens=max_tokens,
        doc_max_words=doc_max_words,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )



@mcp.tool(tags={"search", "necessary"})
def massive_serve_search(
    query: str,
    n_docs: int = 10,
    domains: str = "dpr_wiki_contriever_ivfpq",
    base_url: Optional[str] = None,
    nprobe: Optional[int] = None,
) -> dict:
    """
    Search for documents using massive-serve API for dense passage retrieval.

    This tool provides access to large-scale document collections using dense passage
    retrieval with various embedding models and indices.

    Args:
        query: Search query string
        n_docs: Number of documents to return (default: 10)
        domains: Domain/index to search in (default: "dpr_wiki_contriever_ivfpq")
        base_url: Base URL for the massive-serve API (optional, uses default if not provided)
        nprobe: Number of probes for search (optional, uses API default)

    Returns:
        Dictionary containing search results with the following fields:
        - message: Status message
        - query: The original search query
        - n_docs: Number of documents requested
        - results: Dictionary with IDs, passages, and scores
        - data: Parsed list of search results with passage text, scores, and doc IDs
    """
    # Call the massive-serve API
    response = search_massive_serve(
        query=query,
        n_docs=n_docs,
        domains=domains,
        base_url=base_url,
        nprobe=nprobe,
    )

    # Parse the results for easier consumption
    parsed_results = parse_massive_serve_results(response)

    # Add parsed data to the response for convenience
    response["data"] = [
        {
            "passage": result.passage,
            "score": result.score,
            "doc_id": result.doc_id,
        }
        for result in parsed_results
    ]

    return response


@mcp.tool(tags={"search", "necessary"})
def serper_google_webpage_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
    gl: Annotated[
        str,
        "Geolocation - country code to boost search results whose country of origin matches the parameter value",
    ] = "us",
    hl: Annotated[str, "Host language of user interface"] = "en",
):
    """
    General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources.

    Returns:
        Dictionary containing web search snippets with the following fields:
        - organic: List of organic search results with title, link, and snippet
        - knowledgeGraph: Knowledge graph information (if available)
        - peopleAlsoAsk: List of related questions
        - relatedSearches: List of related searches
    """
    results = search_serper(
        query=query, num_results=num_results, search_type="search", gl=gl, hl=hl
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
def serper_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    include_markdown: Annotated[
        bool, "Whether to include markdown formatting in the response"
    ] = True,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Returns:
        Dictionary containing the webpage content with the following fields:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - url: The original URL that was fetched
        - success: Boolean indicating if the fetch was successful
    """
    try:
        result = fetch_webpage_content(
            url=webpage_url,
            include_markdown=include_markdown,
        )

        result["metadata"] = _sanitize_metadata(result.get("metadata"))
        return {
            **result,
            "success": True,
        }
    except Exception as e:
        return {
            "text": "",
            "markdown": "",
            "metadata": {},
            "url": webpage_url,
            "success": False,
            "error": str(e),
        }


@mcp.tool(tags={"browse", "necessary"})
def serper_fetch_webpages_batch(
    webpage_urls: Annotated[List[str], "List of webpage URLs to fetch in parallel"],
    include_markdown: Annotated[
        bool, "Whether to include markdown formatting in the response"
    ] = True,
    max_workers: Annotated[
        int, "Maximum number of concurrent threads for fetching"
    ] = 30,
) -> List[WebpageContentResponse]:
    """
    Fetch multiple webpages in parallel using ThreadPoolExecutor.
    This is more efficient than calling serper_fetch_webpage_content multiple times
    because it uses a single MCP call for all URLs.

    Returns:
        List of dictionaries, each containing the webpage content with the same fields
        as serper_fetch_webpage_content (text, markdown, metadata, url, success).
    """
    import concurrent.futures

    def fetch_one(url: str) -> dict:
        try:
            # PDF URLs are not handled well by serper's webpage scraper, so we
            # fall back to Jina's reader for those and normalize the response
            # back into the serper WebpageContentResponse schema so callers
            # don't need to special-case PDFs.
            
            result = fetch_webpage_content(
                url=url,
                include_markdown=include_markdown,
            )
            result["metadata"] = _sanitize_metadata(result.get("metadata"))
            return {**result, "url": url, "success": True}
        except Exception as e:
            return {
                "text": "",
                "markdown": "",
                "metadata": {},
                "url": url,
                "success": False,
                "error": str(e),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_one, webpage_urls))

    return results


@mcp.tool(tags={"browse"})
def jina_fetch_webpage_content(
    webpage_url: Annotated[str, "The URL of the webpage to fetch"],
    timeout: Annotated[int, "Request timeout in seconds"] = 30,
) -> JinaWebpageResponse:
    """
    Fetch the content of a webpage using Jina Reader API with timeout support.

    Args:
        webpage_url: The URL of the webpage to fetch
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing the webpage content with the following fields:
        - url: The original URL that was fetched
        - title: Page title
        - content: The webpage content as clean text/markdown
        - description: Page description (if available)
        - publishedTime: Published time (if available)
        - metadata: Additional metadata (lang, viewport, etc.)
        - success: Boolean indicating if the fetch was successful
        - error: Error message if fetch failed
    """
    result = fetch_webpage_content_jina(url=webpage_url, timeout=timeout)
    if isinstance(result, dict) and "metadata" in result:
        result["metadata"] = _sanitize_metadata(result.get("metadata"))
    return result


@mcp.tool(tags={"browse"})
def jina_fetch_webpages_batch(
    webpage_urls: Annotated[List[str], "List of webpage URLs to fetch in parallel"],
    timeout: Annotated[int, "Per-request timeout in seconds"] = 30,
    max_workers: Annotated[
        int, "Maximum number of concurrent threads for fetching"
    ] = 30,
) -> List[JinaWebpageResponse]:
    """
    Fetch multiple webpages in parallel using ThreadPoolExecutor.
    This is more efficient than calling jina_fetch_webpage_content multiple times
    because it uses a single MCP call for all URLs.

    Returns:
        List of dictionaries, each containing the webpage content with the same fields
        as jina_fetch_webpage_content (url, title, content, description, publishedTime,
        metadata, success, error).
    """
    import concurrent.futures

    def fetch_one(url: str) -> dict:
        try:
            result = fetch_webpage_content_jina(url=url, timeout=timeout)
            if isinstance(result, dict) and "metadata" in result:
                result["metadata"] = _sanitize_metadata(result.get("metadata"))
            return {**result, "url": url, "success": True}
        except Exception as e:
            return {
                "url": url,
                "title": "",
                "content": "",
                "description": "",
                "publishedTime": "",
                "metadata": {},
                "success": False,
                "error": str(e),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_one, webpage_urls))

    return results


@mcp.tool(tags={"search", "necessary"})
def serper_google_scholar_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> ScholarResponse:
    """
    Search for academic papers using google scholar (based on Serper.dev API).

    Returns:
        Dictionary containing search results with the following fields:
        - organic: List of organic search results
    """
    results = search_serper_scholar(
        query=query,
        num_results=num_results,
    )

    return results


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_fetch_webpage_content(
    url: Annotated[str, "URL to fetch and extract content from"],
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4AiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4AiResult with extracted webpage content including markdown-formatted text
    """

    from dr_agent.mcp_backend.local.crawl4ai_fetcher import fetch_markdown

    result = await fetch_markdown(
        url=url,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        bypass_cache=bypass_cache,
        headless=True,
        timeout_ms=timeout_ms,
        include_html=include_html,
    )
    return result


@mcp.tool(tags={"browse", "necessary"})
async def crawl4ai_docker_fetch_webpage_content(
    url: Annotated[str, "Target URL to crawl and extract content from"],
    base_url: Annotated[
        Optional[str],
        "Base URL for the Crawl4AI Docker API (e.g., 'http://localhost:8000')",
    ] = None,
    api_key: Annotated[Optional[str], "API key for authentication"] = None,
    use_ai2_config: Annotated[
        bool,
        "If True, use AI2 bot configuration with blocklist (requires CRAWL4AI_BLOCKLIST_PATH env var)",
    ] = False,
    bypass_cache: Annotated[bool, "If True, bypass Crawl4AI cache"] = True,
    ignore_links: Annotated[bool, "If True, remove hyperlinks in markdown"] = True,
    use_pruning: Annotated[
        bool,
        "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
    ] = False,
    bm25_query: Annotated[
        Optional[str],
        "Optional query to enable BM25-based content filtering for focused extraction",
    ] = None,
    timeout_ms: Annotated[int, "Per-page timeout in milliseconds"] = 80000,
    include_html: Annotated[
        bool, "Whether to include raw HTML in the response"
    ] = False,
) -> Crawl4aiApiResult:
    """
    Open a specific URL and extract readable page text as snippets using Crawl4AI Docker API.

    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.

    Returns:
        Crawl4aiApiResult with url, success, markdown-formatted text, and optional fit_markdown/html/error fields
    """
    from dr_agent.mcp_backend.apis.crawl4ai_docker_api import crawl_url_docker

    result = await crawl_url_docker(
        url=url,
        base_url=base_url,
        api_key=api_key,
        bypass_cache=bypass_cache,
        include_html=include_html,
        use_ai2_config=use_ai2_config,
        query=bm25_query,
        ignore_links=ignore_links,
        use_pruning=use_pruning,
        timeout_ms=timeout_ms,
    )
    return result


@mcp.tool(tags={"browse"})
def webthinker_fetch_webpage_content(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import extract_text_from_url

    text = extract_text_from_url(
        url=url,
        snippet=snippet,
        keep_links=keep_links,
    )

    return {"url": url, "text": text}


@mcp.tool(tags={"browse"})
async def webthinker_fetch_webpage_content_async(
    url: str,
    snippet: Optional[str] = None,
    keep_links: bool = False,
) -> dict:
    """
    Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.

    Args:
        url: URL to extract text from
        snippet: Optional snippet to search for and extract context around
        keep_links: Whether to preserve links in the extracted text (default: False)

    Returns:
        Dictionary containing the URL and extracted text content
    """
    from dr_agent.mcp_backend.local.webparsers.webthinker import (
        extract_text_from_url_async,
    )

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=240)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/58.0.3029.110 Safari/537.36",
        "Referer": "https://www.google.com/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, headers=headers
    ) as session:
        text = await extract_text_from_url_async(
            url=url,
            session=session,
            snippet=snippet,
            keep_links=keep_links,
        )

    return {"url": url, "text": text}


@mcp.tool(tags={"search", "local"})
def local_search(
    query: Annotated[str, "Search query string"],
    num_results: Annotated[int, "Number of results to return"] = 10,
) -> LocalSearchResponse:
    """
    Perform a search on a local knowledge source.
    Useful for retrieving relevant passages from specific datasets or local indices.
    """
    if local_searcher is None:
        return {
            "results": [],
            "error": "Local searcher not initialized."
        }
    
    try:
        response = local_searcher.search(query, k=num_results)
        
        if snippet_max_tokens > 0 and snippet_tokenizer:
            for cand in response.get("results", []):
                snippet_text = cand["snippet"]
                tokens = snippet_tokenizer.encode(snippet_text, add_special_tokens=False)
                if len(tokens) > snippet_max_tokens:
                    truncated_tokens = tokens[:snippet_max_tokens]
                    cand["snippet"] = snippet_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return response
    except Exception as e:
        logger.error(f"Error during local search: {e}")
        return {"results": [], "error": str(e)}


@mcp.tool(tags={"browse", "local"})
def local_browse(
    url: Annotated[str, "URL to fetch and extract content from"],
) -> dict:
    """
    Retrieve full text content for a given URL from the local knowledge source.
    Useful for reading the full content of documents found via local_search.
    """
    if local_searcher is None:
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": "Local index not initialized."
        }
    
    try:
        text = local_searcher.get_text_by_url(url)
        if text is None:
            return {
                "url": url,
                "success": False,
                "markdown": "",
                "error": f"URL not found: {url}"
            }
            
        return {
            "url": url,
            "success": True,
            "markdown": text
        }
    except Exception as e:
        logger.error(f"Error during local browse: {e}")
        return {
            "url": url,
            "success": False,
            "markdown": "",
            "error": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport",
        type=str,
        default="http",
        choices=["stdio", "http", "sse", "streamable-http"],
        help="Transport protocol to use (default: stdio for local, http for web)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (for HTTP transports)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (for HTTP transports)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="/mcp",
        help="Path for the HTTP endpoint (default: /mcp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level for the server",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable API response caching",
    )
    parser.add_argument(
        "--local-searcher-type",
        type=str,
        default=None,
        choices=SearcherType.get_choices(),
        help="Type of local searcher to use (default: None, no local search)",
    )
    parser.add_argument(
        "--local-search-max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens for local search snippets (default: 100, set to 0 to disable truncation)",
    )

    # We need a first pass to get the searcher type to add its specific args
    temp_args, _ = parser.parse_known_args()
    if temp_args.local_searcher_type:
        searcher_cls = SearcherType.get_searcher_class(temp_args.local_searcher_type)
        searcher_cls.parse_args(parser)

    args = parser.parse_args()

    # Initialize local searcher if specified
    if args.local_searcher_type:
        try:
            searcher_cls = SearcherType.get_searcher_class(args.local_searcher_type)
            local_searcher = searcher_cls(args)
            logger.info(f"Initialized local searcher: {args.local_searcher_type}")
            
            snippet_max_tokens = args.local_search_max_tokens
            if snippet_max_tokens > 0:
                logger.info(f"Loading tokenizer for local search truncation (max {snippet_max_tokens} tokens)...")
                snippet_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        except Exception as e:
            logger.error(f"Failed to initialize local searcher: {e}")

    # Set cache enabled/disabled based on argument
    if args.no_cache:
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run the server with the provided arguments
    if args.transport == "stdio":
        # stdio transport doesn't accept host/port/path arguments
        # For stdio, we can omit the transport argument since it's the default
        mcp.run(transport="stdio")
    else:
        # HTTP-based transports accept host/port/path/log_level arguments
        mcp.run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
        )
