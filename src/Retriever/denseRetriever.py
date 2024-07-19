import os, pickle
from typing import Optional, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.Retriever.abstractRetriever import AbstractRetriever
from src.Embedder.LMEmbedder import LMEmbedder
from src.Entity.aspect import Aspect
from src.Entity.query import Query


class DenseRetriever(AbstractRetriever):
    """
    Concrete DenseRetriever class.
    """
    cls_type = "dense"
    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, chunks_dir: str, output_path: str, num_chunks: Optional[int] = None, percentile: Optional[float] = None, threshold: Optional[float] = None):
        super().__init__(model=model, query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, num_chunks=num_chunks, percentile=percentile, threshold=threshold)
        self.dense_embedding_dir = embedding_dir
    
    def load_dest_embeddings(self) ->  dict[str, np.ndarray]:
        dests_embs = dict()
        pkls = [f for f in os.listdir(self.dense_embedding_dir)]

        for i in range(0, len(pkls)):
                # if the file's name ends with .pkl, then it is an embeddings file
            if pkls[i].endswith(".pkl"):
                city_name = pkls[i].split("_")[0]
                dests_embs[city_name] = pickle.load(open(f"{self.dense_embedding_dir}/{pkls[i]}", "rb"))
        
        return dests_embs

    def load_data(self) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
        """
        """
        dest_chunks, _ = super().load_data()
        dest_embs = self.load_dest_embeddings()
        return dest_chunks, dest_embs

    def retrieval_for_dest(self, aspects: list[Aspect], dest_emb: np.ndarray, dest_chunks: list[str], chunk_method: Callable) -> dict[str, tuple[float, list[str]]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_emb: dict[str, np.ndarray], dictionary of destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        # return format: {"aspect": (score, top_chunk)}
        dest_result = dict()
        for a in aspects:
            a_text = a.get_new_description()
            description_emb = self.model.encode(a_text)  # shape [1, emb_size]
            score = cosine_similarity(dest_emb, description_emb).flatten()  # shape [chunk_size]
            top_idx = chunk_method(score)
            top_score = score[top_idx]

            # Check if top_score is not empty and does not contain NaN values
            if top_score.size > 0 and not np.isnan(top_score).any():
                avg_score = np.sum(top_score) / top_score.shape[0]  # a scalar score
            else:
                avg_score = 0  # or some default value, depending on your use case

            # retrieve top chunks
            chunks = np.array(dest_chunks)  # [chunk_size]
            top_chunks = chunks[top_idx].tolist()

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result