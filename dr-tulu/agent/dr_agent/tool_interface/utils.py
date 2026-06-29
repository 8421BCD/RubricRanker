"""
Utility functions for MCP agents, including string processing and snippet localization.
"""

import re
from typing import List, Tuple


# Regex patterns for common noise elements in web pages
_NOISE_PATTERNS = [
    re.compile(r"(Cookie|Privacy)\s*(Policy|Notice|Settings).*", re.IGNORECASE),
    re.compile(r"(Subscribe|Sign\s*up|Log\s*in|Register)\s*(now|here|to).*", re.IGNORECASE),
    re.compile(r"(Advertisement|Sponsored|Promoted)\b.*", re.IGNORECASE),
    re.compile(r"©\s*\d{4}.*?(?:All\s*rights?\s*reserved)?.*", re.IGNORECASE),
    re.compile(r"(Skip to|Jump to)\s+(main |primary )?content.*", re.IGNORECASE),
]


def clean_webpage_text_old(raw_text: str) -> str:
    """Rule-based cleaning: remove common noise patterns from webpage text."""
    if not raw_text:
        return ""
    lines = raw_text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are just navigation / noise
        if any(pat.match(stripped) for pat in _NOISE_PATTERNS):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)




# Matches any run of digits — works for "[2] > [1]", "2 > 1", "[2] [1] [3]", etc.
_DIGIT_PATTERN = re.compile(r"\d+")


def parse_listwise_ranking(raw_ranking: str, num_documents: int) -> List[int]:
    """Parse a listwise LLM ranking output into 0-based document indices.

    Assumes the raw output is some variant of a ranked list of 1-based indices,
    e.g. ``"[2] > [1] > [3]"``, ``"2 > 1 > 3"`` or ``"[2] [1] [3]"``. All digit
    runs are extracted in order; non-digit characters (brackets, ``>``, spaces,
    etc.) are ignored.

    Out-of-range indices are dropped, duplicates are removed while preserving
    first-seen order, and any documents not mentioned by the LLM are appended at
    the end in their original order so no document is silently lost.

    Args:
        raw_ranking: The raw text output from the listwise reranker LLM.
        num_documents: Total number of candidate documents passed to the LLM.

    Returns:
        A list of 0-based document indices of length ``num_documents`` describing
        the reranked order. Returns an empty list if no indices can be parsed.
    """
    if not raw_ranking or num_documents <= 0:
        return []

    # Extract every digit run in order (handles "[i]", "i", whitespace/`>` separators).
    ordered_indices: List[int] = []
    seen = set()
    for idx_str in _DIGIT_PATTERN.findall(raw_ranking):
        idx = int(idx_str) - 1  # 1-based -> 0-based
        if 0 <= idx < num_documents and idx not in seen:
            seen.add(idx)
            ordered_indices.append(idx)

    if not ordered_indices:
        return []

    # Append any documents the LLM did not rank, in their original order, so that
    # we never silently drop candidates.
    # for idx in range(num_documents):
    #     if idx not in seen:
    #         ordered_indices.append(idx)

    return ordered_indices

# Import for sentence tokenization with fallback
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    # Fallback if nltk is not available
    def sent_tokenize(text: str) -> List[str]:
        return re.split(r"(?<=[.!?]) +", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from text for better matching"""
    return re.sub(r"[^\w\s]", " ", text)


def f1_score(set1: set, set2: set) -> float:
    """Calculate F1 score between two sets of words"""
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    if intersection == 0:
        return 0.0

    precision = intersection / len(set1)
    recall = intersection / len(set2)

    return 2 * precision * recall / (precision + recall)


def extract_snippet_with_context(
    full_text: str, snippet: str, context_chars: int = 3000
) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful,
                         the second element is the extracted context.
    """
    try:
        # Limit full text to prevent excessive processing
        full_text = full_text[:100000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2  # Minimum threshold

        sentences = sent_tokenize(full_text)

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first part of the full text
            return False, full_text[: context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


# ---------------------------------------------------------------------------
# Webpage / Markdown text cleaning
# ---------------------------------------------------------------------------
#
# Serper's scrape API returns markdown-formatted text when includeMarkdown=True
# (see dr_agent/mcp_backend/apis/serper_apis.py::fetch_webpage_content and
# dr_agent/tool_interface/mcp_tools.py::SerperBrowseTool, which prefers the
# ``markdown`` field over ``text``). As a result, the raw text we clean here
# is expected to be *markdown*, not HTML. The rules below operate directly on
# markdown syntax (links, images, headings, lists, tables, code fences, etc.)
# and only strip residual HTML tags as a best-effort fallback.

# --- Line-level noise patterns (template / boilerplate lines to drop) -------
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*(Cookie|Privacy)\s*(Policy|Notice|Settings)\b.*$", re.IGNORECASE),
    re.compile(
        r"^\s*(Subscribe|Sign\s*up|Log\s*in|Register)\s*(now|here|to)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(Advertisement|Sponsored|Promoted)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}.*$", re.IGNORECASE),
    re.compile(r"^\s*(Skip to|Jump to)\s+(main |primary )?content\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(Share|Tweet|Email|Print)\s+this\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Related\s+articles?\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Back\s+to\s+top\b.*$", re.IGNORECASE),
    # Horizontal rules in markdown
    re.compile(r"^\s*([-*_])\1{2,}\s*$"),
    # Markdown reference-link definitions: [ref]: http://...
    re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$"),
    # Lone URL on its own line
    re.compile(r"^\s*https?://\S+\s*$"),
    # Template html-block tags left over (rare)
    re.compile(r"^\s*<(?:script|style|nav|header|footer|form|aside|button|img)\b", re.IGNORECASE),
]

# HTML tags whose *entire content* should be removed wholesale if they ever
# leak into the markdown (e.g. when the scraper couldn't convert them).
_HTML_BLOCK_STRIP_RE = re.compile(
    r"<\s*(script|style|nav|header|footer|form|aside|button|noscript)\b[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Self-closing / inline tags that should be dropped entirely (no textual value).
_HTML_VOID_STRIP_RE = re.compile(
    r"<\s*(img|input|meta|link|br|hr)\b[^>]*/?>",
    re.IGNORECASE,
)

# Any other residual HTML tag: drop the tag itself but keep surrounding text.
_HTML_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*\b[^>]*>")

# Markdown image syntax: ![alt](url "title") — drop entirely (no textual value
# for us beyond the alt text, which is usually decorative).
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Image wrapped in a link: [![alt](img_url)](link_url) — drop entirely.
# Handle this BEFORE the generic link rule below.
_MD_IMG_LINKED_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")

# Markdown inline link: [text](url "optional title") — keep only the visible
# text (recursively so nested brackets like [foo [bar] baz](url) behave OK).
# Note: this pattern is applied repeatedly until no more matches remain.
_MD_LINK_RE = re.compile(r"\[([^\[\]]*)\]\([^)]*\)")

# Markdown reference-style link: [text][ref] or [ref] — keep visible text.
_MD_REFLINK_RE = re.compile(r"\[([^\[\]]+)\]\[[^\]]*\]")


def _strip_markdown_decorations(text: str) -> str:
    """Remove markdown link/image syntax but preserve visible text content."""
    # Drop linked images first (they would otherwise partially match the
    # generic link rule and leave dangling alt text).
    text = _MD_IMG_LINKED_RE.sub("", text)
    # Drop plain images.
    text = _MD_IMAGE_RE.sub("", text)
    # Unwrap inline links: [text](url) -> text. Apply repeatedly to handle
    # nested or adjacent occurrences produced by the above substitutions.
    prev = None
    while prev != text:
        prev = text
        text = _MD_LINK_RE.sub(r"\1", text)
    # Unwrap reference-style links: [text][ref] -> text.
    text = _MD_REFLINK_RE.sub(r"\1", text)
    return text


def _strip_residual_html(text: str) -> str:
    """Best-effort removal of any residual HTML tags in the markdown."""
    # Strip entire <script>...</script> style blocks (content + tags).
    text = _HTML_BLOCK_STRIP_RE.sub("", text)
    # Strip void / content-less tags entirely.
    text = _HTML_VOID_STRIP_RE.sub("", text)
    # Strip any remaining standalone tags but preserve surrounding text.
    text = _HTML_ANY_TAG_RE.sub("", text)
    return text


def _normalize_line(line: str) -> str:
    """Normalise a single line by stripping markdown heading/list/quote markers.

    We want to KEEP the textual content of headings, list items, blockquotes
    and table cells (they carry the real information) while removing the
    structural markdown characters themselves so the reranker sees clean text.
    """
    stripped = line.strip()
    if not stripped:
        return ""

    # Heading: "## Foo" -> "Foo"
    m = re.match(r"^#{1,6}\s+(.*)$", stripped)
    if m:
        return m.group(1).strip()

    # Blockquote: "> quoted" -> "quoted"
    m = re.match(r"^>+\s*(.*)$", stripped)
    if m:
        return m.group(1).strip()

    # Ordered / unordered list item: "- foo" / "* foo" / "1. foo" -> "foo"
    m = re.match(r"^(?:[-*+]|\d+[.)])\s+(.*)$", stripped)
    if m:
        return m.group(1).strip()

    # Markdown table row: "| a | b |" -> "a | b". Skip separator rows like
    # "| --- | --- |" which carry no info.
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(re.match(r"^:?-{3,}:?$", c) for c in cells if c):
            return ""
        return " | ".join(c for c in cells if c)

    return stripped


def clean_markdown_text(raw_text: str) -> str:
    """Clean markdown-formatted webpage text for reranking.

    Keeps the textual content of:
      - paragraphs
      - headings (#..######)
      - ordered & unordered lists
      - blockquotes (>)
      - tables (row cell text)
      - fenced code blocks (``` ... ```) — inside-code text is preserved
      - bold/italic text (markers stripped)

    Removes:
      - markdown images ``![alt](url)`` and linked images ``[![...](...)](...)``
      - markdown links ``[text](url)`` — keeps only ``text``
      - reference-style link definitions ``[ref]: url`` and reference links
      - template / boilerplate lines (cookies, subscribe, ©, nav, etc.)
      - residual ``<script>`` / ``<style>`` / ``<nav>`` / ``<header>`` /
        ``<footer>`` / ``<button>`` / ``<img>`` HTML blocks (and generic tags)
      - horizontal rules, isolated URL-only lines
    """
    if not raw_text:
        return ""

    # 1) Strip whole HTML blocks (script/style/nav/header/footer/…).
    text = _strip_residual_html(raw_text)

    # 2) Strip markdown image & link decorations while keeping visible text.
    text = _strip_markdown_decorations(text)

    # 3) Remove bold/italic markers: **x**, __x__, *x*, _x_ -> x
    #    (run twice so **_x_** style combos fully unwrap)
    for _ in range(2):
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"__(.+?)__", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", text)
        text = re.sub(r"(?<!_)_([^_\n]+?)_(?!_)", r"\1", text)

    # 4) Remove inline code backticks: `x` -> x (fenced code block fences
    #    ``` are just dropped as empty delimiters; their inner content is
    #    preserved line-by-line below).
    text = re.sub(r"`{3,}[^\n]*", "", text)  # ```python etc. -> ""
    text = re.sub(r"`([^`\n]+)`", r"\1", text)

    # 5) Line-level cleanup: drop noise lines, normalise structural markers.
    cleaned: List[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in _NOISE_LINE_PATTERNS):
            continue
        normalised = _normalize_line(raw_line)
        if not normalised:
            continue
        # After all the stripping, a line may have collapsed into just
        # punctuation / a single char — drop those, they carry no info.
        if not re.search(r"\w", normalised):
            continue
        cleaned.append(normalised)

    # 6) Collapse runs of whitespace inside each kept line.
    cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned]

    # 7) Replace any residual square brackets with parentheses so the final
    #    cleaned text cannot collide with the ``[1] [2] [3]`` document
    #    identifiers used by the listwise reranker prompt (otherwise citation
    #    markers like ``[12]`` inside paragraphs can confuse the ranker).
    #    At this point markdown links/images/reference definitions have all
    #    been stripped, so the only brackets left are incidental ones in the
    #    body text and can be safely rewritten.
    cleaned = [line.replace("[", "(").replace("]", ")") for line in cleaned]

    return "\n".join(cleaned)


def clean_webpage_text(raw_text: str, url: str = "") -> str:
    """Entry point: clean webpage scrape output.

    Always runs the markdown cleaner regardless of the source URL (PDFs
    included). We no longer special-case PDFs because the current scrapers
    (e.g. the Tencent Jina proxy) return usable markdown for PDF URLs, and
    the old PDF-specific branch was prone to clearing valid content when the
    markdown happened to contain page-number-like lines.

    The ``url`` parameter is retained for backward compatibility with
    existing call sites but is currently unused.
    """
    # del url  # unused; kept for backward compatibility
    if not raw_text:
        return ""
    return clean_markdown_text(raw_text)
