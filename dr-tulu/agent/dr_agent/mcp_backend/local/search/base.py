import argparse
from abc import ABC, abstractmethod
from typing import List, Optional
from typing_extensions import NotRequired, TypedDict

class LocalSearchResult(TypedDict):
    docid: str
    snippet: str
    url: str
    score: Optional[float]

class LocalSearchResponse(TypedDict):
    results: List[LocalSearchResult]
    error: NotRequired[str]

class BaseSearcher(ABC):
    """Abstract base class for all search implementations."""

    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add searcher-specific arguments to the argument parser."""
        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="Dataset name for document retrieval in local search (default: Tevatron/browsecomp-plus-corpus)",
        )

    @abstractmethod
    def __init__(self, args):
        """Initialize the searcher with parsed arguments."""
        pass

    @abstractmethod
    def search(self, query: str, k: int = 10) -> LocalSearchResponse:
        """
        Perform search and return results.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            LocalSearchResponse containing list of search results.
        """
        pass

    @abstractmethod
    def get_text_by_url(self, url: str) -> Optional[str]:
        """
        Retrieve full text for a given URL from the local index.
        """
        pass

    @property
    @abstractmethod
    def search_type(self) -> str:
        """Return the type of search (e.g., 'BM25', 'FAISS')."""
        pass

    def search_description(self, k: int = 10) -> str:
        """
        Description of the search tool to be passed to the LLM.
        """
        return f"Perform a search on a local knowledge source. Returns top-{k} hits with docid, score, and snippet."
