import json
import os
from typing import Dict, List, Optional, Union

import dotenv
import requests
from typing_extensions import TypedDict

from ..cache import cached

# Load environment variables
dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 20))
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", 20))  # Separate timeout for webpage scraping


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


############### official serper api ###############
# --- Original simple (single-request) official stub, kept for reference ---
# @cached()
# def search_serper(
#     query: str,
#     num_results: int = 10,
#     gl: str = "us",
#     hl: str = "en",
#     search_type: str = "search",  # Can be "search", "places", "news", "images"
#     api_key: str = None,
# ) -> SearchResponse:
#     """
#     Search using Serper.dev API for general web search.
#
#     Args:
#         query: Search query string
#         num_results: Number of results to return (default: 10)
#         gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
#         hl: Host language of user interface (default: en)
#         search_type: Type of search to perform (default: "search")
#                     Options: "search", "places", "news", "images"
#         api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)
#
#     Returns:
#         SearchResponse containing:
#         - searchParameters: Dict with search metadata
#         - knowledgeGraph: Optional knowledge graph information
#         - organic: List of organic search results
#         - peopleAlsoAsk: Optional list of related questions
#         - relatedSearches: Optional list of related search queries
#     """
#     if not api_key:
#         import os
#
#         api_key = os.getenv("SERPER_API_KEY")
#         if not api_key:
#             raise ValueError(
#                 "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
#             )
#
#     url = "https://google.serper.dev/search"
#
#     payload = json.dumps({"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type})
#
#     headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
#
#     try:
#         response = requests.post(url, headers=headers, data=payload)
#
#         if response.status_code != 200:
#             raise Exception(
#                 f"API request failed with status {response.status_code}: {response.text}"
#             )
#
#         return response.json()
#
#     except requests.exceptions.RequestException as e:
#         raise Exception(f"Error performing Serper search: {str(e)}")


# --- Active official implementation: paginated + concurrent ---
# NOTE: The official Serper API (https://google.serper.dev/search) caps a single
# response at 10 organic results regardless of the `num` value sent. To return
# more than 10 results we issue concurrent `page=1,2,...` requests, mirroring

# Maximum number of results per single Serper API request
SERPER_MAX_PER_PAGE = 10

# Official Serper API endpoint
SERPER_OFFICIAL_SEARCH_URL = "https://google.serper.dev/search"


def _search_serper_single_page(
    query: str,
    num_results: int = 10,
    page: int = 1,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",
    api_key: str = None,
    tbs: Optional[str] = None,
    location: Optional[str] = None,
) -> SearchResponse:
    """
    Perform a single-page Serper search request against the official Serper API
    (internal helper).

    Args:
        query: Search query string
        num_results: Number of results requested for this page (effectively
            capped at 10 by the API)
        page: Page number (1-based)
        gl: Country code
        hl: Host language
        search_type: Type of search ("search", "places", "news", "images")
        api_key: Serper API key. The official API uses the `X-API-KEY` header.
        tbs: Time-based search filter
        location: Location string

    Returns:
        Raw SearchResponse from the Serper API
    """
    if not api_key:
        api_key = SERPER_API_KEY
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "q": query,
        "num": num_results,
        "hl": hl,
        "type": search_type,
        "page": page,
    }

    # Use location if provided, otherwise fall back to gl (country code)
    if location:
        payload["location"] = location
    else:
        payload["gl"] = gl

    if tbs:
        payload["tbs"] = tbs

    try:
        response = requests.post(
            SERPER_OFFICIAL_SEARCH_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=TIMEOUT,
        )
        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper search: {str(e)}")


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",
    api_key: str = None,
    tbs: Optional[str] = None,
    location: Optional[str] = None,
) -> SearchResponse:
    """
    Search using the official Serper.dev API for general web search.
    Automatically paginates when num_results > 10, since each request returns
    at most 10 organic results.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10).
            If > 10, multiple paginated requests will be made concurrently.
        gl: Country code to boost search results whose country of origin
            matches the parameter value (default: us). Ignored if location is set.
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        api_key: Serper API key. If not provided, will use SERPER_API_KEY env var.
        tbs: Time-based search filter (e.g. "qdr:h" for past hour,
             "qdr:d" for past day, "qdr:w" for past week)
        location: Location string (e.g. "United States"). When set, gl is ignored.

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results (up to num_results)
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    # Fetch pages concurrently (works for both single and multi-page requests)
    import math
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_pages = math.ceil(num_results / SERPER_MAX_PER_PAGE)

    # Pre-compute per-page request parameters: (page_number, per_page_count)
    page_params = []
    remaining = num_results
    for page in range(1, total_pages + 1):
        per_page = min(SERPER_MAX_PER_PAGE, remaining)
        page_params.append((page, per_page))
        remaining -= per_page

    def _fetch_page(page_num: int, per_page: int):
        """Fetch a single page; returns (page_num, result) or (page_num, None) on error."""
        try:
            result = _search_serper_single_page(
                query=query,
                num_results=per_page,
                page=page_num,
                gl=gl,
                hl=hl,
                search_type=search_type,
                api_key=api_key,
                tbs=tbs,
                location=location,
            )
            return (page_num, result)
        except Exception as e:
            print(f"Warning: Serper pagination page {page_num} failed: {e}")
            return (page_num, None)

    # Fire all page requests concurrently
    page_results = {}
    with ThreadPoolExecutor(max_workers=total_pages) as executor:
        futures = {
            executor.submit(_fetch_page, pnum, psize): pnum
            for pnum, psize in page_params
        }
        for future in as_completed(futures):
            page_num, result = future.result()
            if result is not None:
                page_results[page_num] = result

    if not page_results:
        raise Exception("Error performing Serper search: all page requests failed")

    # Merge results in page order
    all_organic = []
    merged_response = None

    for page_num in sorted(page_results.keys()):
        page_result = page_results[page_num]

        # Use the first page's response as the base for metadata
        if merged_response is None:
            merged_response = page_result
        else:
            # Merge additional metadata from later pages if present
            if "peopleAlsoAsk" in page_result and "peopleAlsoAsk" not in merged_response:
                merged_response["peopleAlsoAsk"] = page_result["peopleAlsoAsk"]
            if "relatedSearches" in page_result and "relatedSearches" not in merged_response:
                merged_response["relatedSearches"] = page_result["relatedSearches"]

        # Collect organic results
        page_organic = page_result.get("organic", [])
        all_organic.extend(page_organic)

    # Replace organic results with the merged list (trimmed to num_results)
    merged_response["organic"] = all_organic[:num_results]

    return merged_response


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
) -> ScholarResponse:
    """
    Search academic papers using Serper.dev Scholar API.

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    if not api_key:
        import os

        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query, "num": num_results})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        return response.json()

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error performing Serper scholar search: {str(e)}")


############### official serper scrape api ###############
# Official Serper scrape endpoint
SERPER_OFFICIAL_SCRAPE_URL = "https://scrape.serper.dev"


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using the official Serper.dev scrape API.

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key. If not provided, will use SERPER_API_KEY env var.

    Returns:
        WebpageContentResponse containing:
        - url: The original URL that was scraped (injected for convenience)
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
        - credits: Number of credits consumed by this request
    """
    if not api_key:
        api_key = SERPER_API_KEY
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})

    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    try:
        # Timeout prevents slow pages from blocking the entire pipeline
        response = requests.post(
            SERPER_OFFICIAL_SCRAPE_URL,
            headers=headers,
            data=payload,
            timeout=SCRAPE_TIMEOUT,
        )

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        data = response.json()
        # The official API response does not include the source URL; inject it so
        # downstream consumers can always rely on data["url"].
        data["url"] = url
        return data

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching webpage content: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing API response: {str(e)}")




# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
