import logging
import os
from typing import List, Optional

from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher

from .base import BaseSearcher, LocalSearchResponse, LocalSearchResult

logger = logging.getLogger(__name__)


class BM25Searcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        super().parse_args(parser)
        parser.add_argument(
            "--index-path",
            required=True,
            help="Name or path of Lucene index (e.g. msmarco-v1-passage).",
        )

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if not args.index_path:
            raise ValueError("index_path is required for BM25 searcher")

        self.searcher = None
        self.docid_to_metadata = None
        self.url_to_text = None

        logger.info(f"Initializing BM25 searcher with index: {args.index_path}")

        try:
            self.searcher = LuceneSearcher(args.index_path)
        except Exception as exc:
            raise ValueError(
                f"Index '{args.index_path}' is not a valid local Lucene index path."
            ) from exc

        self._load_dataset()

        logger.info("BM25 searcher initialized successfully")

    def _load_dataset(self) -> None:
        logger.info(f"Loading dataset: {self.args.dataset_name}")

        try:
            dataset_cache = os.getenv("HF_DATASETS_CACHE")
            if dataset_cache:
                cache_dir = dataset_cache
            else:
                cache_dir = None

            if self.args.dataset_name.endswith(".jsonl"):
                ds = load_dataset(
                    "json",
                    data_files=self.args.dataset_name,
                    split="train",
                    cache_dir=cache_dir,
                )
            else:
                ds = load_dataset(
                    self.args.dataset_name, split="train", cache_dir=cache_dir
                )
            # Store both text and url
            self.docid_to_metadata = {
                str(row["id"]): {"snippet": row["contents"], "url": row["url"]}
                for row in ds
            }
            # Create a URL to text mapping for local_browse
            self.url_to_text = {row["url"]: row["contents"] for row in ds}
            logger.info(f"Loaded {len(self.docid_to_metadata)} passages from dataset")
        except Exception as e:
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                logger.error(
                    f"Dataset '{self.args.dataset_name}' access failed. This is likely an authentication issue."
                )
                logger.error("Possible solutions:")
                logger.error("1. Ensure you are logged in to Hugging Face:")
                logger.error("   huggingface-cli login")
                logger.error("2. Set environment variable:")
                logger.error("   export HF_TOKEN=your_token_here")
                logger.error(
                    "3. Check if the dataset name is correct and you have access"
                )
                logger.error(f"Current environment variables:")
                logger.error(
                    f"   HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}"
                )
                logger.error(
                    f"   HUGGINGFACE_HUB_TOKEN: {'Set' if os.getenv('HUGGINGFACE_HUB_TOKEN') else 'Not set'}"
                )

                try:
                    from huggingface_hub import HfApi

                    api = HfApi()
                    user_info = api.whoami()
                    logger.error(
                        f"   Hugging Face user: {user_info.get('name', 'Unknown')}"
                    )
                except Exception as auth_e:
                    logger.error(
                        f"   Hugging Face authentication check failed: {auth_e}"
                    )

            raise RuntimeError(
                f"Failed to load dataset '{self.args.dataset_name}': {e}"
            )

    def search(self, query: str, k: int = 10) -> LocalSearchResponse:
        if not self.searcher:
            raise RuntimeError("Searcher not initialized")

        hits = self.searcher.search(query, k)
        results: List[LocalSearchResult] = []

        for hit in hits:
            metadata = self.docid_to_metadata[hit.docid]
            results.append(
                {
                    "docid": hit.docid,
                    "score": float(hit.score),
                    "snippet": metadata["snippet"],
                    "url": metadata["url"],
                }
            )
        return {"results": results}

    def get_text_by_url(self, url: str) -> Optional[str]:
        if self.url_to_text is None:
            return None
        return self.url_to_text.get(url)

    @property
    def search_type(self) -> str:
        return "BM25"
