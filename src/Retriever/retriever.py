import os
import pickle
import json
import numpy as np
from tqdm import tqdm

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *
from src.Query.query import *


class Retriever:
    """
    Retriever class
    """
    def __init__(self, model: LMEmbedder):
        """
        Retriever class
        :param queries:
        :param model:
        """
        self.model = model
    
    def load_queries(self, query_path: str, raw_query: bool = True) -> list[str] | list[Query]:
        """
        """
        if raw_query:
            with open(query_path, "r") as file:
                file.readline()
                queries = [line.strip() for line in file]
        else:
            with open(query_path, "rb") as file:
                queries = pickle.load(file)
        return queries
    
    def load_dest_embeddings(self, embs_dir: str) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
        """
        Load embeddings
        """
        dests_chunks = dict()
        dests_embs = dict()
        pkls = sorted([f for f in os.listdir(embs_dir)])
        for i in range(0, len(pkls)):
            # if the file's nameends with chunks.pkl, then it is a chunks file
            if pkls[i].endswith("chunks.pkl"):
                city_name = pkls[i].split("_")[0]
                dests_chunks[city_name] = pickle.load(open(f"{embs_dir}/{pkls[i]}", "rb"))
            # if the file's name ends with emb.pkl, then it is an embeddings file
            elif pkls[i].endswith("emb.pkl"):
                city_name = pkls[i].split("_")[0]
                dests_embs[city_name] = pickle.load(open(f"{embs_dir}/{pkls[i]}", "rb"))
        return dests_chunks, dests_embs

    def calculate_similarity_score(self, query: str | list[str], dests: dict[str, np.ndarray]) -> dict[str, dict[str, np.array]]:
        """
        Calculate similarity score between query and embedding
        :param query:
        :param emb:
        """
        queries = [query] if isinstance(query, str) else query
        scores = dict()
        for q in queries:
            query_emb = self.model.encode(q)
            # Scores is a dictionary with key as query and value as a dictionary with key as dest_name and value as score
            scores[q] = dict()
            for dest_name, dest_emb in dests.items():
                # dest_emb is a list of embeddings in sections
                # query_emb is a single embedding
                score = np.dot(dest_emb, query_emb.T).squeeze() / (np.linalg.norm(dest_emb, axis=1) * np.linalg.norm(query_emb)).squeeze()
                scores[q][dest_name] = score
        return scores
    
    # def extract_descriptions(self, query: Query | list[Query]) -> list[list[str]]:
    #     """
    #     Extract descriptions
    #     """
    #     # for step 
    #     query_descriptions = []
    #     queries = [query] if isinstance(query, Query) else query
    #     for q in queries:
    #         descriptions = q.get_descriptions()
    #         query_descriptions.append(descriptions)
    #     return query_descriptions
        
    def dense_retrieval(self, query: str | list[str], dests: dict[str, np.ndarray], percentile: float) -> dict[str, dict[str, tuple[float, list[int]]]]:
        """
        # Retrieve associated embeddings
        """
        scores = self.calculate_similarity_score(query, dests)
        dense_results = dict()
        for q in scores.keys():
            dense_results[q] = dict()
            for dest_name, score in scores[q].items():
                # Sort the scores in descending order
                sorted_score = np.argsort(score)
                # Put the average of tatal sections' score and top sentences in the dense_results
                # dense_results[q][dest_name] = (sum(score[:int(percentile * score.shape[0])]) / len(score[:int(percentile * score.shape[0])]), 
                #                                sorted_score[:int(percentile * score.shape[0])])
        return dense_results
    
    def run_dense_retrieval(self, query_path: str, embs_dir: str, percentile: float, output_dir: str) -> None:
        """
        Run dense retrieval
        """
        
        queries = self.load_queries(query_path=query_path, raw_query=True) # a list of string
        dests_chunks, dests_embs = self.load_dest_embeddings(embs_dir)
        dense_results = self.dense_retrieval(queries, dests_embs, percentile)
        # final_dese_results is a dictionary with key as query and value as a dictionary with key as city_name and value as a tuple with total_score and top_sections
        final_dense_results = dict()
        # Find each query's top sections
        for q in dense_results.keys():
            final_dense_results[q] = dict()
            for dest_name in dense_results[q].keys():
                top_sections = []
                score, section_idx = dense_results[q][dest_name]
                for idx in section_idx:
                    top_sections.append(dests_chunks[dest_name][idx])
                final_dense_results[q][dest_name] = (score, top_sections)
        # save the final_dense_results into a json file
        output_file = os.path.join(output_dir, "dense_results.json")
        with open(output_file, "w") as file:
            json.dump(dense_results, file)

    # def run_dense_retrieval_with_qe(self, query_path: str, embs_dir: str, percentile: float, output_dir: str) -> None:
    #     """
    #     Run dense retrieval
    #     """
    #     # load query description
    #     queries = self.load_queries(query_path=query_path, raw_query=False) # a list of Query
    #     descriptions = []
    #     for q in queries:
    #         d = q.get_descriptions()
    #         descriptions.append(d)

    #     dests_chunks, dests_embs = self.load_dest_embeddings(embs_dir)

    #     # run dense retrieval for each constraint groups
    #     for query in descriptions:
    #         for constraint in query:
    #             dense_results = self.dense_retrieval(constraint, dests_embs, percentile)
                
    #             # aggregate method: mean
    #             for c, c_results in dense_results.items():
    #                 pass
                
    #     # final_dese_results is a dictionary with key as query and value as a dictionary with key as city_name and value as a tuple with total_score and top_sections
    #     final_dense_results = dict()
    #     # Find each query's top sections
    #     for q in dense_results.keys():
    #         final_dense_results[q] = dict()
    #         for dest_name in dense_results[q].keys():
    #             top_sections = []
    #             score, section_idx = dense_results[q][dest_name]
    #             for idx in section_idx:
    #                 top_sections.append(dests_chunks[dest_name][idx])
    #             final_dense_results[q][dest_name] = (score, top_sections)
    #     pickle.dump(dense_results, open(output_dir, "wb"))

    #     pass
                

    # def run_dense_retrieval(self, percentile: float, index_dir: str, save=True, output_dir=None,
    #                         load_if_exists=True, path_suffix=None, query_embeds=None):
    #     """
    #     Run dense retrieval
    #     :param percentile:
    #     :param index_dir:
    #     :param save:
    #     :param output_dir:
    #     :param load_if_exists:
    #     :param path_suffix:
    #     :param query_embeds:
    #     """
    #     # Create output directory if it does not exist
    #     full_query_path = os.path.join(output_dir, "query_dense_results_full.pkl")
    #     query_path = os.path.join(output_dir, f"query_dense_results_full_{path_suffix}.pkl") if path_suffix \
    #         else full_query_path
    #     score_path = os.path.join(output_dir,
    #                               f"query_dense_results_full_scores_{path_suffix}.pkl") if path_suffix \
    #         else os.path.join(output_dir, "query_dense_results_full_scores.pkl")

    #     # Load embeddings
    #     query_dense_results = {}
    #     pkls = sorted([f for f in os.listdir(index_dir)])

    #     skip = False
    #     query_scores = None

    #     if load_if_exists and os.path.exists(query_path):

    #         query_dense_results = pickle.load(open(query_path, "rb"))
    #         query_scores = pickle.load(open(score_path, "rb"))
    #         # load full path as well and combine
    #         if os.path.exists(full_query_path):
    #             query_dense_results_full = pickle.load(open(full_query_path, "rb"))
    #             for query in query_dense_results_full:
    #                 if query not in query_dense_results:
    #                     query_dense_results[query] = query_dense_results_full[query]
    #         skip = True

    #     percentile_thresholds = []
    #     query_scores_dict = {}

    #     for m, query in enumerate(self.queries):
    #         if skip and query in query_scores:
    #             percentile_thresholds.append(np.percentile(query_scores[query], percentile))
    #             continue
    #         query_scores_vals = []
    #         if query_embeds is not None:
    #             query_emb = query_embeds[m]
    #         else:
    #             query_emb = self.model.encode(query)
    #         for i in range(0, len(pkls), 2):
    #             emb_file = pkls[i]
    #             sent_file = pkls[i + 1]
    #             city_name = emb_file.split("_")[0]
    #             emb = pickle.load(open(f"{index_dir}/{emb_file}", "rb"))
    #             # Use cosine similarity instead of dot product
    #             curr_scores = np.dot(emb, query_emb.T) / (np.linalg.norm(emb) * np.linalg.norm(query_emb))
    #             query_scores_vals += curr_scores.tolist()
    #             del emb
    #         query_scores_vals = np.array(query_scores_vals)
    #         query_scores_dict[query] = query_scores_vals
    #         percentile_thresholds.append(np.percentile(query_scores_vals, percentile))
    #         if save:
    #             # save query scores
    #             with open(score_path, "wb") as f:
    #                 pickle.dump(query_scores_dict, f)
    #             del query_scores_dict

    #         if not skip or query not in query_dense_results:
    #             for j, query in tqdm(enumerate(self.queries)):
    #                 dense_results = {}

    #                 if query_embeds is not None:
    #                     query_emb = query_embeds[m]
    #                 else:
    #                     query_emb = self.model.encode(query)
    #                 for i in range(0, len(pkls), 2):
    #                     emb_file = pkls[i]
    #                     sent_file = pkls[i + 1]
    #                     city_name = emb_file.split("_")[0]

    #                     emb = pickle.load(open(f"{index_dir}/{emb_file}", "rb"))
    #                     sentences = pickle.load(open(f"{index_dir}/{sent_file}", "rb"))

    #                     # get score for each sentence
    #                     scores = []
    #                     for s in emb:
    #                         # Use cosine similarity instead of dot product
    #                         score = np.dot(s, query_emb) / (np.linalg.norm(s) * np.linalg.norm(query_emb))
    #                         scores.append(score)

    #                     sent_score_tuples = list(zip(sentences, scores))
    #                     sent_score_tuples.sort(key=lambda x: x[1], reverse=True)

    #                     dense_results[city_name] = sent_score_tuples

    #                 query_dense_results[query] = dense_results

    #         if save:
    #             pickle.dump(query_dense_results, open(query_path, "wb"))

    #         # print(list(query_dense_results.keys()))

    #     out_query_dense_results = {}

    #     for j, query in enumerate(self.queries):
    #         out_dense_results = {}

    #         for i in range(0, len(pkls), 2):
    #             emb_file = pkls[i]
    #             sent_file = pkls[i + 1]
    #             city_name = emb_file.split("_")[0]

    #             # get sentences above threshold
    #             top_sentences = []
    #             agg_score = 0
    #             for sent, score in query_dense_results[query][city_name]:
    #                 if score >= percentile_thresholds[j]:
    #                     top_sentences.append(sent)
    #                     agg_score += score

    #             out_dense_results[city_name] = (top_sentences, agg_score)

    #         sorted_dense_results = sorted(out_dense_results.items(), key=lambda x: x[1][1], reverse=True)

    #         out_query_dense_results[query] = sorted_dense_results
    #     del query_dense_results

    #     return out_query_dense_results
