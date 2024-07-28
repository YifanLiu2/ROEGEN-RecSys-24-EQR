# import json, abc
# from tqdm import tqdm
# from rank_bm25 import BM25Okapi

# from src.LLM.GPTChatCompletion import LLM, GPTChatCompletion
# from src.Retriever.abstractRetriever import *
# from src.Retriever.sparseRetriever import *
# from src.Retriever.denseRetriever import *
# from src.Embedder.LMEmbedder import LMEmbedder
# from src.LLM.GPTChatCompletion import *
# from src.Retriever.sparseRetriever import GPTChatCompletion
# from config import API_KEY

# class ProQERetriever(DenseRetriever):
#     """
#     """
#     cls_type = "dense"
#     def __init__(self, query_path: str, output_dir: str, chunks_dir: str, embedding_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), model: LMEmbedder = None, num_chunks: tuple[int] = (3, 10), power: int = 5, num_iter: int = 5, beta: int = 1, gemma: int = 0):
#         super().__init__(query_path=query_path, output_dir=output_dir, chunks_dir=chunks_dir, embedding_dir=embedding_dir, model=model, num_chunks=num_chunks, power=power)
#         self.llm = llm
#         self.num_iter = num_iter
#         self.beta = beta
#         self.gemma = gemma 

#     def retrieval_for_query(self, query: Query | str, dests_chunks: dict[str, list[str]], dests_embs: dict[str, np.ndarray] = None) -> dict[str, tuple[float, dict[str, list[str]]]]: 
#         """
#         Retrieval for a single query for ProQeRetriever.
#         Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
#         """
#         # return format: {dest: (dest_score, {"aspect": top_chunk})}
#         final_results = dict()
#         query = query
#         visited = set()

#         # iterative PRF over num_iter
#         for i in range(self.num_iter): 

#             # retrieve the pseudo document for all apsects 
#             query_results = dict()

#             if isinstance(query, str):
#                 aspects = [Aspect(query)]
#             else:
#                 aspects = query.get_all_aspects()
                
#             for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
#                 num_chunks = 1 if i < self.num_iter - 1 else None # only retrieve the 1st chunk except for the last iteration
#                 dest_result = self.retrieval_for_dest(aspects=aspects, dest_chunks=dests_chunks[dest_name], dest_emb=dest_emb, num_chunks=num_chunks)
#                 query_results[dest_name] = dest_result

#             # rank the score for each aspect
#             ranked_results = dict()
#             for dest, dest_result in query_results.items():
#                 for aspect, (score, top_chunk) in dest_result.items():
#                     if aspect not in ranked_results:
#                         ranked_results[aspect] = []
#                     ranked_results[aspect].append((dest, score, top_chunk))

#             for aspect in ranked_results:
#                 ranked_results[aspect].sort(key=lambda x: x[1], reverse=True)
            
#             keywords_map = dict()

#             # retrieve the pseudo document for one aspect
#             for aspect in aspects:
#                 aspect_text = aspect.get_new_description()

#                 # find the top 1 chunks for the aspect
#                 top_dest = None
#                 for i, dest in enumerate(ranked_results[aspect_text]):
#                     if dest[0] not in visited:
#                         top_dest = i
#                         visited.add(dest[0])
#                         break

#                 # check relevance
#                 relevance = 0
#                 chunk = ranked_results[aspect_text][top_dest][2][0]
#                 is_relevant = self.get_llm_relevance(chunk, aspect_text)
#                 keywords = self.get_gpt_keywords(aspect_text, curr_passage=chunk)

#                 if is_relevant:
#                     relevance = 1

#                 # update weight and aspect
#                 self.update_weight(keywords, keywords_map, relevance)
#                 new_aspect_text = self.update_aspect(aspect_text=aspect_text, keywords_map=keywords_map)
#                 aspect.set_new_description(new_aspect_text)

#         for dest_name, dest_result in query_results.items(): # for the last query results
#             dest_result = self.avg_fusion(dest_result)
#             final_results[dest_name] = dest_result
            
#         return final_results
    

#     def get_llm_relevance(self, curr_passage: str, original_aspect: str) -> bool:
#         prompt = """
#         Given a user's travel cities recommendation query, determine whether the provided passage from a city's introduction serve as good evidence for the query.
        
#         Answer only in "yes" or "no".
#         passage: {curr_passage}
#         query: {original_aspect}
#         Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
#         """
#         message = [
#             {"role": "system", "content": "You are a travel expert."},
#             {"role": "user", "content": prompt.format(curr_passage=curr_passage, original_aspect=original_aspect)},
#         ]
#         response = self.llm.generate(message)
#         # parse the answer
#         start = response.find("{")
#         end = response.rfind("}") + 1
#         response = response[start:end]
#         response = json.loads(response)["answer"]

#         if 'yes' in response.lower():
#             return True
#         return False
    

#     def get_gpt_keywords(self, aspect_text: str, curr_passage: str) -> list[str]:
#         prompt = """
#         Given a initial query and a related passage, identify a list of keywords from the passage that once added to the initial query, could be used to better retrieve passages that contain the actual answer. 
#         Format your answer in the following JSON format: {{"answer": [KEY WORD LIST]}}, with key being "answer" and value being a python list.
#         query: {query}
#         passage: {passage}
#         keywords:
#         """
#         prompt = prompt.format(query=aspect_text, passage=curr_passage)
#         message = [
#             {"role": "system", "content": "You are a travel expert."},
#             {"role": "user", "content": prompt},
#         ]
#         response = self.llm.generate(message)

#         # parse the answer
#         start = response.find("{")
#         end = response.rfind("}") + 1
#         response = response[start:end]
#         keywords = json.loads(response)["answer"]

#         return keywords


#     def update_weight(self, keywords_list: list[str], keywords_map: dict[str, list[int]], relevance: int):
#         for key in keywords_list:
#             key = key.lower().strip()
#             if key not in keywords_map:
#                 keywords_map[key] = []
#                 if relevance == 1:
#                     keywords_map[key].append([1, relevance])
#                 else:
#                     keywords_map[key].append([0, relevance])
#             else:
#                 if relevance == 1:
#                     keywords_map[key][0][0] += self.beta
#                     keywords_map[key][0][1] = relevance
#                 else:
#                     keywords_map[key][0][0] -= self.gemma
#                     keywords_map[key][0][1] = keywords_map[key][0][1] or relevance
    
#     def update_aspect(self, aspect_text: str, keywords_map: dict[str, list[int]]) -> str:
#         for key in keywords_map:
#             times = int(keywords_map[key][0][0]) # weight
#             for _ in range(times): # how many times to append that keyword to the query
#                 aspect_text += " " + key
        
#         return aspect_text




    


