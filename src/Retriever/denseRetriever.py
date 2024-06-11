import os, pickle, json, abc
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *
from src.Query.query import *

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
    def retrieval(self, queries: list, dests_emb: dict[str, np.ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict:
        """
        Abstract method to be implemented for dense retrieval.

        :param queries: list, the list of queries to process.
        :param dests_emb: dict[str, np.ndarray], dictionary mapping destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary mapping destination names to their textual chunks.
        :param percentile: float, the percentile value to determine the threshold for filtering top results.
        :return: dict, the structured results of retrieval formatted by queries and destinations.
        """
        pass
        
    def run_retrieval(self) -> None:
        """
        Loads necessary data and runs the dense / dense + sparse retrieval process, then saves the results to the specified output path.
        """
        # load data
        queries = self.load_queries() # list[str]
        dests_chunks, dests_embs = self.load_dest_embeddings()

        # run dense retrieval
        final_results = self.retrieval(queries=queries, dests_emb=dests_embs, dests_chunks=dests_chunks, percentile=self.percentile)
        
        # save results
        with open(self.output_path, "w") as file:
            json.dump(final_results, file, indent=4)


class DenseRetriever(AbstractRetriever):
    """
    Concrete DenseRetriever class.
    """
    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, output_path: str, percentile: float = 10):
        # check query path
        if not query_path.endswith(".txt"):
            raise ValueError(f"Invalid query path: '{query_path}'. Must be a txt file.")
        super().__init__(model, query_path, embedding_dir, output_path, percentile)
    
    def retrieval(self, queries: list[str], dests_emb: dict[str, np.ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict[str, dict[str, tuple[float, list[str]]]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_emb: dict[str, np.ndarray], dictionary of destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        # return format: {"query": {"dest": (dest_score, top_chunk)}}
        descriptions = queries
        dense_results = dict()
        # for each query
        for d in descriptions:
            print("-----------------------------")
            print(f"Process query: {d}")
            description_emb = self.model.encode(d) # shape [1, emb_size]
            dense_results[d] = dict()
            # for each destination, calculate similarity score
            for dest_name, dest_emb in dests_emb.items(): # shape [chunk_size, emb_size]
                score = cosine_similarity(dest_emb, description_emb).flatten() # shape [chunk_size]
                # threshold = np.percentile(score, percentile) # determine the threshold
                
                # extract top idx and top score
                # top_idx = np.where(score >= threshold)[0]
                # top_score = score[score >= threshold]

                # top_score is the top 3 score
                top_idx = np.argsort(score)[-3:]
                top_score = score[top_idx]
                avg_score = np.sum(top_score) / top_score.shape[0] # a scalar score

                # retrieve top chunks
                chunks = np.array(dests_chunks[dest_name]) # [chunk_size]
                top_chunks = chunks[top_idx].tolist()
    
                # store results
                dense_results[d][dest_name] = (avg_score, top_chunks)
        return dense_results

class DenseRetrieverQE(AbstractRetriever):
    """
    Extended the DenseRetriever to support query expansion.
    """
    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, output_path: str, percentile: float = 10):
        super().__init__(model, query_path, embedding_dir, output_path, percentile)

    def retrieval(self, queries: list[Query], dests_emb: dict[str, np.ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict[str, dict[str, tuple[float, dict[str, list[str]]]]]:
        """
        Processes each query with its subqueries (or constraints) and associated weights for a weighted dense retrieval.

        :param queries: list[Query], queries with potential expansions.
        :param dests_emb: dict[str, np.ndarray], dictionary of destination names to their embeddings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict, detailed retrieval results with scores and relevant chunks for each subquery.
        """
        # return format: {"query": {"dest": (dest_score, {"subquery": top_chunk})}}

        dense_results = dict()
        
        # for each query
        for q in queries:
            print("-----------------------------")
            print(f"Process query: {q.description}")
            dense_results[q.description] = dict()
            descriptions = q.get_descriptions()
            weights = q.get_description_weights()
            assert len(descriptions) == len(weights) # one to one map between description and weight

            # for each destination
            for dest_name, dest_emb in tqdm(dests_emb.items(), desc="Processing destinations"): # shape [chunk_size, emb_size]
                # assume aggregate method is sum
                dest_score = 0
                sub_dense_results = dict()  # results for subquery
                # for each subquery
                for d, w in zip(descriptions, weights):
                    description_emb = self.model.encode(d) # shape [1, emb_size]
                    score = cosine_similarity(dest_emb, description_emb).flatten() # shape [chunk_size]
                    # threshold = np.percentile(score, percentile) # determine the threshold
                    
                    # extract top idx and top score
                    # top_idx = np.where(score >= threshold)[0]
                    # top_score = score[score >= threshold]

                    # top_score is the top 3 score
                    top_idx = np.argsort(score)[-3:]
                    top_score = score[top_idx]
                    avg_score = np.sum(top_score) / top_score.shape[0] # a scalar score

                    # retrieve top chunks
                    chunks = np.array(dests_chunks[dest_name]) # [chunk_size]
                    top_chunks = chunks[top_idx].tolist()
        
                    # store subquery results
                    sub_dense_results[d] = top_chunks
                    dest_score += avg_score * w # avg score weighted by w
            
                # store dest results
                dense_results[q.description][dest_name] = (dest_score, sub_dense_results)

        return dense_results
    
class DenseRetrieverElaboration(AbstractRetriever):
    """
    Extended the DenseRetriever to support query elaboration.
    """

    def __init__(self, model: LMEmbedder, query_path: str, embedding_dir: str, output_path: str, percentile: float = 10):
        super().__init__(model, query_path, embedding_dir, output_path, percentile)

    def retrieval(self, queries: list[Query], dests_emb: dict[str, np.ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict[str, dict[str, tuple[float, dict[str, list[str]]]]]:

        dense_results = dict()
        
        # for each query
        for q in queries:
            print("-----------------------------")
            print(f"Process query: {q.description}")
            dense_results[q.description] = dict()
            constraints = q.constraints
            descriptions = constraints.get_descriptions()
            weights = q.get_description_weights()
            assert len(descriptions) == len(weights) # one to one map between description and weight

            # for each destination
            for dest_name, dest_emb in tqdm(dests_emb.items(), desc="Processing destinations"):
                dest_score = 0
                sub_dense_results = dict()  # results for subquery
                # for each subquery (description)
                for d, w in zip(descriptions, weights):
                    description_emb = self.model.encode(d)
                    score = cosine_similarity(dest_emb, description_emb).flatten()

                    # top_score is the top 3 score
                    top_idx = np.argsort(score)[-3:]
                    top_score = score[top_idx]
                    avg_score = np.sum(top_score) / top_score.shape[0]

                    chunks = np.array(dests_chunks[dest_name])
                    top_chunks = chunks[top_idx].tolist()

                    # store subquery results
                    sub_dense_results[d] = top_chunks
                    dest_score += avg_score * w
                    
                # store dest results
                dense_results[q.description][dest_name] = (dest_score, sub_dense_results)

        return dense_results
                                                                                                                                               
                                                                                                                                               