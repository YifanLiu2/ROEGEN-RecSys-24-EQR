import os, pickle, json, abc
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *
from src.Entity.query import *

class AbstractRetriever(abc.ABC):
    """
    Abstract base dense retriever class.
    """
    def __init__(self, model: LMEmbedder, query_path: str, dense_embedding_dir: str, output_path: str, percentile: float = 10):
        self.model = model
        self.query_path = query_path
        
        # check embedding dir
        if not os.path.exists(dense_embedding_dir):
            raise ValueError(f"Invalid directory path for destination embeddings: {dense_embedding_dir}")
        self.dense_embedding_dir = dense_embedding_dir

        # check output path
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = output_path

        self.percentile = percentile
    
    def load_queries(self) -> list[str] | list[Query]:
        """
        Loads queries from the specified file path. Queries can be in plain text format or serialized objects.
        :return: list[str] | list[Query], a list of queries, either as strings or Query objects depending on the file format.
        """
        if self.query_path.endswith("txt"):
            with open(self.query_path, "r") as file:
                queries = [line.strip() for line in file]
        else: # pickle file
            with open(self.query_path, "rb") as file:
                queries = pickle.load(file)
        return queries

    def load_dest_embeddings(self) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Loads destination text chunks and associated embeddings from the specified directory.
        :return: tuple(dict[str, list[str]], dict[str, np.ndarray], dict[str, np.ndarray] | None), a tuple containing dictionaries for destination chunks and embeddings.
        """
        dests_chunks = dict()
        dests_embs = dict()
        pkls = sorted([f for f in os.listdir(self.dense_embedding_dir)])
        for i in range(0, len(pkls)):
            # if the file's name ends with chunks.pkl, then it is a chunks file
            if pkls[i].endswith("chunks.pkl"):
                city_name = pkls[i].split("_")[0]
                dests_chunks[city_name] = pickle.load(open(f"{self.dense_embedding_dir}/{pkls[i]}", "rb"))
            # if the file's name ends with emb.pkl, then it is an embeddings file
            elif pkls[i].endswith("emb.pkl"):
                city_name = pkls[i].split("_")[0]
                dests_embs[city_name] = pickle.load(open(f"{self.dense_embedding_dir}/{pkls[i]}", "rb"))

        return dests_chunks, dests_embs

    @abc.abstractmethod
    def retrieval_for_dest(self, queries: list, dests_emb: dict[str, np.ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict:
        """
        Abstract method to be implemented for dense retrieval.

        :param queries: list, the list of queries to process.
        :param dests_emb: dict[str, np.ndarray], dictionary mapping destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary mapping destination names to their textual chunks.
        :param percentile: float, the percentile value to determine the threshold for filtering top results.
        :return: dict, the structured results of retrieval formatted by queries and destinations.
        """
        pass

    def run_retrieval(self):
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        queries = self.load_queries()
        dests_chunks, dests_embs = self.load_dest_embeddings()

        results = dict()
        for query in queries:
            query_results = self.retrieval_for_query(query, dests_embs, dests_chunks)
            if isinstance(query, Query):
                query = query.get_description()

            results[query] = query_results

        with open(self.output_path, "w") as file:
            json.dump(results, file, indent=4)

    
    def fusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
        """
        Fusion method to combine results from multiple aspects of a destination.

        :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
        :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
        """
        # return format: (dest_score, {"aspect": top_chunk})
        fused_results = tuple()
        dest_score = 0
        top_chunks = dict()
        for aspect, (score, chunks) in dest_results.items():
            dest_score += score
            top_chunks[aspect] = chunks
        # count the average score
        dest_score /= len(dest_results)
        fused_results = (dest_score, top_chunks)
        return fused_results
        

    def retrieval_for_query(self, query: Query, dests_embs: dict[str, np.ndarray], dests_chunks: dict[str, list[str]]) -> dict[str, tuple[float, dict[str, list[str]]]]: 
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        # return format: {dest: (dest_score, {"aspect": top_chunk})}

        query_results = dict()

        # retrieve results for each destination
        for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
            aspects = query.get_all_aspects()
            dest_result = self.retrieval_for_dest(aspects, dest_emb, dests_chunks[dest_name], self.percentile)
            # fuse results from multiple aspects
            dest_result = self.fusion(dest_result)
            query_results[dest_name] = dest_result

        return query_results 


class DenseRetriever(AbstractRetriever):
    """
    Concrete DenseRetriever class.
    """
    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, output_path: str, percentile: float = 10):
        super().__init__(model, query_path, embedding_dir, output_path, percentile)
    

    def retrieval_for_dest(self, aspects: list[Aspect], dest_emb: np.ndarray, dest_chunks: list[str], percentile: float) -> dict[str, tuple[float, list[str]]]:
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
            description_emb = self.model.encode(a_text) # shape [1, emb_size]
            score = cosine_similarity(dest_emb, description_emb).flatten() # shape [chunk_size]
            # threshold = np.percentile(score, percentile) # determine the threshold

            # extract top idx and top score with threshold
            # top_idx = np.where(score >= threshold)[0]
            # top_score = score[score >= threshold]

            # extract top 3 idx and top 3 score
            top_idx = np.argsort(score)[-3:]
            top_score = score[top_idx]
            avg_score = np.sum(top_score) / top_score.shape[0] # a scalar score

            # retrieve top chunks
            chunks = np.array(dest_chunks) # [chunk_size]
            top_chunks = chunks[top_idx].tolist()

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result