from src.Retriever.denseRetriever import *
from rank_bm25 import BM25Okapi
from src.Entity.query import AbstractQuery


class SparseRetriever(AbstractRetriever):
    """
    Concrete SparseRetriever class.
    """
    cls_type = "sparse"

class BM25Retriever(SparseRetriever):
    """
    Concrete SparseRetriever class.
    """
    def retrieval_for_dest(self, query: AbstractQuery, dest_chunks: list[str]) -> dict[str, tuple[float, list[str]]]:
        corpus = dest_chunks
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        num_chunks = self.num_chunks

        # compute bm25 scores for a_text and dest_chunks      
        reformulation = query.get_reformulation()
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_r = reformulation.split(" ")

        # get the score 
        score = bm25.get_scores(tokenized_r)
        top_idx = np.argsort(score)[-num_chunks:]
        top_score = score[top_idx]
        avg_score = self.calculate_city_score(top_score)

        # retrieve top chunks
        chunks = np.array(dest_chunks) # [chunk_size]
        top_chunks = chunks[top_idx].tolist()

        return avg_score, top_chunks

       
