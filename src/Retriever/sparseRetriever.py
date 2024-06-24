from src.Retriever.denseRetriever import *
from rank_bm25 import BM25Okapi
from sparsembed import model, retrieve
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.Entity.aspect import Aspect

class SparseRetriever(AbstractRetriever):
    """
    Concrete SparseRetriever class.
    """
    cls_type = "sparse"
    def __init__(self, query_path: str, chunks_dir: str, output_path: str, percentile: float = 10):
        super().__init__(query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, percentile=percentile)


class BM25Retriever(SparseRetriever):
    """
    Concrete DenseRetriever class.
    """
    def __init__(self, query_path: str, chunks_dir:str, output_path: str, percentile: float = 10):
        super().__init__(model=None, query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, percentile=percentile)
    

    def retrieval_for_dest(self, aspects: list[Aspect], dest_chunks: list[str], percentile: float) -> dict[str, tuple[float, list[str]]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        # return format: {"aspect": (score, top_chunk)}
        dest_result = dict()
        corpus = dest_chunks
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        for a in aspects:
            a_text = a.get_new_description()
            # compute bm25 scores for a_text and dest_chunks
            tokenized_a = a_text.split(" ")

            top_chunks = bm25.get_top_n(tokenized_a, corpus, n=3)
            # get the score for the top chunks
            scores = bm25.get_scores(tokenized_a)
            scores.sort()

            avg_score = sum(scores[-3:]) / 3

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result
    

class SpladeRetriever(SparseRetriever):
    """
    Concrete SparseRetriever class.
    """

    def __init__(self, query_path: str, chunks_dir: str, output_path: str, percentile: float = 10, batchsize: int = 8):
        super().__init__(query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, percentile=percentile)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = model.Splade(
            model=AutoModelForMaskedLM.from_pretrained("naver/splade_v2_max").to(self.device),
            tokenizer=AutoTokenizer.from_pretrained("naver/splade_v2_max"),
            device=self.device
        )
        self.batch_size = batchsize
    
    def retrieval_for_dest(self, aspects: list[Aspect], dest_chunks: list[str], percentile: float) -> dict[str, tuple[float, list[str]]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        dest_result = dict()
        aspects_text = [a.get_new_description() for a in aspects]
        docs = [{"id": chunk, "chunks": chunk} for chunk in dest_chunks]
        retriever = retrieve.SpladeRetriever(
            key="id", # Key identifier of each document.
            on="chunks", # Fields to search.
            model=self.model # Splade retriever.
        )

        retriever = retriever.add(
            documents=docs,
            batch_size=self.batch_size,
            k_tokens=256, # Number of activated tokens.
        )

        splade_results = retriever(
            aspects_text, # Aspect
            k_tokens=20, # Maximum number of activated tokens.
            k = 3, # Number of top-k documents to retrieve.
            batch_size=self.batch_size
        )
        
        count = 0
        for result in splade_results:
            top_chunks = []
            score =[]
            for r in result:
                top_chunks.append(r["id"])
                score.append(r["similarity"])
            avg_score = sum(score) / len(score)
            dest_result[aspects_text[count]] = (avg_score, top_chunks)
            count += 1
        return dest_result

    