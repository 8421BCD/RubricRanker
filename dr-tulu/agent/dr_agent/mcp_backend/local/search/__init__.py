from enum import Enum

try:
    from .bm25_retriever import BM25Searcher
except ImportError:
    BM25Searcher = None

try:
    from .faiss_retriever import FaissSearcher
except ImportError:
    FaissSearcher = None

class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""
    BM25 = ("bm25", BM25Searcher)
    FAISS = ("faiss", FaissSearcher)
    
    def __init__(self, cli_name, searcher_class):
        self.cli_name = cli_name
        self.searcher_class = searcher_class
    
    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]
    
    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                if searcher_type.searcher_class is None:
                    raise ImportError(
                        f"Searcher type '{cli_name}' requires additional dependencies "
                        f"(e.g., tevatron, qwen_omni_utils). Please install them first."
                    )
                return searcher_type.searcher_class
        raise ValueError(f"Unknown searcher type: {cli_name}")

__all__ = ["BaseSearcher", "FaissSearcher", "BM25Searcher", "SearcherType"]
