import os, pickle
from typing import Optional, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.Retriever.abstractRetriever import AbstractRetriever
from src.Embedder.LMEmbedder import LMEmbedder
from src.Entity.query import *


class DenseRetriever(AbstractRetriever):
    """
    Concrete DenseRetriever class.
    """
    cls_type = "dense"
    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, chunks_dir: str, output_dir: str, num_chunks: int = 10):
        super().__init__(model=model, query_path=query_path, output_dir=output_dir, chunks_dir=chunks_dir, num_chunks=num_chunks)
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

    def retrieval_for_dest(self, query: AbstractQuery | np.ndarray, dest_emb: np.ndarray, dest_chunks: list[str]) -> tuple[float, list[str]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_emb: dict[str, np.ndarray], dictionary of destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        # embed query reformulation
        if isinstance(query, AbstractQuery):
            description_emb = self.model.encode(query.get_reformulation())  # shape [1, emb_size]
        else:
            description_emb = query

        # calculate city score
        score = cosine_similarity(dest_emb, description_emb).flatten()  # shape [chunk_size]
        top_idx = np.argsort(score)[-self.num_chunks:]
        top_score = score[top_idx]
        avg_score = self.calculate_city_score(top_score)

        # retrieve top chunks
        chunks = np.array(dest_chunks)  # [chunk_size]
        top_chunks = chunks[top_idx].tolist()

        return avg_score, top_chunks
    