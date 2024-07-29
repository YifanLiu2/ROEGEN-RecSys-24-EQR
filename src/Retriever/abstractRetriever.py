import os, pickle, json, abc
from typing import Optional, Callable
import numpy as np
from tqdm import tqdm
from src.Entity.query import AbstractQuery
from src.Embedder.LMEmbedder import LMEmbedder

class AbstractRetriever(abc.ABC):
    """
    Abstract base dense retriever class.
    """
    def __init__(self, query_path: str, output_dir: str, chunks_dir: str,  model: Optional[LMEmbedder] = None, num_chunks: tuple[int] = (10, 3), power: int = 5):
        self.model = model
        self.query_path = query_path
        self.output_dir = output_dir
        self.chunks_dir = chunks_dir

        # set up hyper param
        self.num_chunks = num_chunks
        self.power = power
        
    
    def load_queries(self) -> list[str] | list[AbstractQuery]:
        """
        Loads queries from the specified file path. Queries can be in plain text format or serialized objects.
        :return: list[str] | list[AbstractQuery], a list of queries, either as strings or Query objects depending on the file format.
        """
        if self.query_path.endswith("txt"):
            with open(self.query_path, "r") as file:
                queries = [line.strip() for line in file]
        else: # pickle file
            with open(self.query_path, "rb") as file:
                queries = pickle.load(file)
        return queries
    
    def load_chunks(self) -> dict[str, list[str]]:
        """
        """
        dests_chunks = dict()
        pkls = [f for f in os.listdir(self.chunks_dir)]
        for i in range(0, len(pkls)):
            # if the file's name ends with .pkl, then it is a chunks file
            if pkls[i].endswith(".pkl"):
                city_name = pkls[i].split("_")[0]
                dests_chunks[city_name] = pickle.load(open(f"{self.chunks_dir}/{pkls[i]}", "rb"))
        return dests_chunks

    def load_data(self) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
        """
        """
        return self.load_chunks(), None

    @abc.abstractmethod
    def retrieval_for_dest(self, query: AbstractQuery, dest_chunks: dict[str, list[str]], dest_emb: dict[str, np.ndarray] = None) -> tuple[float, list[str]]:
        """
        Abstract method to be implemented for dense retrieval.

        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param dests_emb: dict[str, np.ndarray], dictionary of destination names to their embeddings.
        :return: dict, structured results with scores and top matching chunks.
        """
        pass

    def run_retrieval(self):
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        queries = self.load_queries()
        dests_chunks, dests_embs = self.load_data()

        results = dict()
        for query in queries:
            query_results = self.retrieval_for_query(query=query, dests_embs=dests_embs, dests_chunks=dests_chunks)
            query_str = query.get_description()
            results[query_str] = query_results

        dense_result_path = os.path.join(self.output_dir, "dense_result.json")
        ranked_list_path = os.path.join(self.output_dir, "ranked_list.json")
        
        # save dense result
        with open(dense_result_path, "w") as file_dense:
            json.dump(results, file_dense, indent=4)
        
        with open(ranked_list_path, "w") as file_ranked:
            ranked_list = {
                query: [dest for dest in results[query]] 
                for query in results
            }
            json.dump(ranked_list, file_ranked, indent=4)

    
    def retrieval_for_query(self, query: AbstractQuery, dests_embs: dict[str, np.ndarray], dests_chunks: dict[str, list[str]] = None) -> dict[str, tuple[float, list[str]]]: 
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        # return format: {dest: (dest_score, top_chunks)}
        query_results = dict()

        if self.cls_type == "sparse": # sparse retrieval
            for dest_name, dest_chunks in tqdm(dests_chunks.items(), desc="Processing destinations"):
                avg_score, top_chunks = self.retrieval_for_dest(query=query, dest_chunks=dest_chunks)
                query_results[dest_name] = (avg_score, top_chunks)

        else: # dense retrieval
            # retrieve results for each destination
            for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
                avg_score, top_chunks = self.retrieval_for_dest(query=query, dest_chunks=dests_chunks[dest_name], dest_emb=dest_emb)
                query_results[dest_name] = (avg_score, top_chunks)

        sorted_query_results = dict(sorted(query_results.items(), key=lambda item: item[1][0], reverse=True))
        return sorted_query_results 
    

    def calculate_city_score(self, top_score: np.array) -> float:
        if top_score.size == 0:
            return 0  

        scores = (top_score + 1) ** self.power
        return float(np.mean(scores))



    ############# FUSION ################
    # def avg_fusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
    #     """
    #     Fuse results from multiple aspects using average score.

    #     :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
    #     :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
    #     """
    #     # return format: (dest_score, {"aspect": top_chunk})
        
    #     fused_results = tuple()
    #     dest_score = 0
    #     top_chunks = dict()
    #     for aspect, (score, chunks) in dest_results.items():
    #         dest_score += score
    #         top_chunks[aspect] = chunks
    #     # count the average score
    #     dest_score /= len(dest_results)
    #     fused_results = (dest_score, top_chunks)
    #     return fused_results
    
    # def gfusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
    #     """
    #     Fuse results from multiple aspects using geometric mean.

    #     :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
    #     :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
    #     """
    #     # return format: (dest_score, {"aspect": top_chunk})
        
    #     fused_results = tuple()
    #     dest_score = 1
    #     top_chunks = dict()
    #     for aspect, (score, chunks) in dest_results.items():
    #         dest_score *= score
    #         top_chunks[aspect] = chunks
    #     dest_score = dest_score**(1/len(dest_results))
    #     # geometrical avg
    #     fused_results = (dest_score, top_chunks)
    #     return fused_results
      
    # def hfusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
    #     """
    #     Fuse results from multiple aspects using harmonic mean.

    #     :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
    #     :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
    #     """
    #     # return format: (dest_score, {"aspect": top_chunk})
        
    #     fused_results = tuple()
    #     dest_score = 0
    #     top_chunks = dict()
    #     for aspect, (score, chunks) in dest_results.items():
    #         dest_score += (1/score)
    #         top_chunks[aspect] = chunks
    #     dest_score = len(dest_results)/dest_score
    #     # harmonic avg
    #     fused_results = (dest_score, top_chunks)
    #     return fused_results