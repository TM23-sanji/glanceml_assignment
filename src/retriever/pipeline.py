import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.retriever.query_parser import QueryParser
from src.retriever.search import FashionSearcher
from src.models.schemas import RetrievalResult


class RetrievalPipeline:
    def __init__(self, processed_dir="data/processed"):
        print("Loading query parser...")
        self.parser = QueryParser()
        print("Loading search index...")
        self.searcher = FashionSearcher(processed_dir)

    def query(self, text: str, k: int = 10):
        parsed = self.parser.parse(text)
        results = self.searcher.search(text, parsed, top_n=200)

        top_k = results["ranked"][:k]
        output = []
        for rank, (idx, final, clip_sim, atomic_sim, attr_con) in enumerate(top_k):
            output.append(
                RetrievalResult(
                    path=str(self.searcher.image_paths[idx]),
                    score=float(final),
                    clip_similarity=float(clip_sim),
                    atomic_similarity=float(atomic_sim),
                    attr_consistency=float(attr_con),
                )
            )

        return parsed, output
