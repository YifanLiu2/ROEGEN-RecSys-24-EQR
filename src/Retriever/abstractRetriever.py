import os, pickle, json, abc
import numpy as np
from tqdm import tqdm
from src.Entity.query import Query
from src.Embedder.LMEmbedder import LMEmbedder
from src.Entity.aspect import Aspect

class AbstractRetriever(abc.ABC):
    """
    Abstract base dense retriever class.
    """
    def __init__(self, query_path: str, output_path: str, chunks_dir: str, num_chunks: int = 3, model: LMEmbedder = None):
        self.model = model
        self.query_path = query_path
        self.output_path = output_path
        self.num_chunks = num_chunks
        self.chunks_dir = chunks_dir
    
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
    def retrieval_for_dest(self, aspects: list, dest_chunks: dict[str, list[str]], num_chunks: int = 3, dest_emb: dict[str, np.ndarray] = None) -> dict:
        """
        Abstract method to be implemented for dense retrieval.

        :param aspects: list[Aspect], list of aspects to retrieve for.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
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
            if isinstance(query, Query):
                query = query.get_description()

            results[query] = query_results

        with open(self.output_path, "w") as file:
            json.dump(results, file, indent=4)
    
    def retrieval_for_query(self, query: Query | str, dests_embs: dict[str, np.ndarray], dests_chunks: dict[str, list[str]] = None) -> dict[str, tuple[float, dict[str, list[str]]]]: 
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        # return format: {dest: (dest_score, {"aspect": top_chunk})}

        query_results = dict()
        if isinstance(query, str):
            aspects = [Aspect(query)]
        else:
            aspects = query.get_all_aspects()
        
        if self.cls_type == "sparse": # sparse retrieval
            for dest_name, dest_chunks in tqdm(dests_chunks.items(), desc="Processing destinations"):
                dest_result = self.retrieval_for_dest(aspects=aspects,dest_chunks=dest_chunks, num_chunks=self.num_chunks)
                # fuse results from multiple aspects
                dest_result = self.avg_fusion(dest_result)
                query_results[dest_name] = dest_result

        else: # dense retrieval
            # retrieve results for each destination
            for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
                dest_result = self.retrieval_for_dest(aspects=aspects, dest_chunks=dests_chunks[dest_name], num_chunks=self.num_chunks, dests_emb=dest_emb)
                # fuse results from multiple aspects
                dest_result = self.avg_fusion(dest_result)
                query_results[dest_name] = dest_result

        return query_results 
    




    ############## FUSION ################
    def avg_fusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
        """
        Fuse results from multiple aspects using average score.

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
    
    def gfusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
        """
        Fuse results from multiple aspects using geometric mean.

        :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
        :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
        """
        # return format: (dest_score, {"aspect": top_chunk})
        
        fused_results = tuple()
        dest_score = 1
        top_chunks = dict()
        for aspect, (score, chunks) in dest_results.items():
            dest_score *= score
            top_chunks[aspect] = chunks
        dest_score = dest_score**(1/len(dest_results))
        # geometrical avg
        fused_results = (dest_score, top_chunks)
        return fused_results
      
    def hfusion(self, dest_results: dict[str, tuple[float, list[str]]]) -> tuple[float, dict[str, list[str]]]:
        """
        Fuse results from multiple aspects using harmonic mean.

        :param dest_results: dict[str, tuple[float, list[str]]], results from each destination.
        :return: dict[str, tuple[float, list[str]]], fused results from all destinations.
        """
        # return format: (dest_score, {"aspect": top_chunk})
        
        fused_results = tuple()
        dest_score = 0
        top_chunks = dict()
        for aspect, (score, chunks) in dest_results.items():
            dest_score += (1/score)
            top_chunks[aspect] = chunks
        dest_score = len(dest_results)/dest_score
        # harmonic avg
        fused_results = (dest_score, top_chunks)
        return fused_results