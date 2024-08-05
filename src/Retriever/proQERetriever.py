import json, abc
from tqdm import tqdm

from src.LLM.GPTChatCompletion import LLM, GPTChatCompletion
from src.Retriever.abstractRetriever import *
from src.Retriever.sparseRetriever import *
from src.Retriever.denseRetriever import *
from src.Embedder.LMEmbedder import LMEmbedder
from src.LLM.GPTChatCompletion import *
from config import API_KEY

class ProQERetriever(DenseRetriever):
    """
    """
    cls_type = "dense"
    def __init__(self, query_path: str, output_dir: str, chunks_dir: str, embedding_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), model: LMEmbedder = None, num_chunks: int = 10, num_iter: int = 5, beta: int = 1, gemma: int = 0):
        super().__init__(query_path=query_path, output_dir=output_dir, chunks_dir=chunks_dir, embedding_dir=embedding_dir, model=model, num_chunks=num_chunks)
        self.llm = llm
        self.num_iter = num_iter
        self.beta = beta
        self.gemma = gemma 

    def retrieval_for_query(self, query: AbstractQuery, dests_chunks: dict[str, list[str]], dests_embs: dict[str, np.ndarray] = None) -> tuple[float, list[str]]:
        """
        Retrieval for a single query for ProQeRetriever.
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        # init var for prf
        visited = set()
        chunk_embs_prf = np.concatenate(list(dests_embs.values())) # prepare the dest embs for prf
        dest_chunks_prf = [item for lst in dests_chunks.values() for item in lst]
        query_str = query.get_description()
        query_emb = self.model.encode(query_str) 
        
        # iterative PRF over num_iter
        for _ in tqdm(range(self.num_iter), desc="Iterative PRF Processing"):

            # prf iteration: get top chunk
            keywords_map = dict()
            sort_prf_chunks = self.sort_prf_chunks(query_emb=query_emb, chunk_embs=chunk_embs_prf, dest_chunks=dest_chunks_prf)
            top_chunk = None
            for chunk in sort_prf_chunks:
                if chunk not in visited:
                    visited.add(chunk)
                    top_chunk = chunk
                    break
            
            # prf iteration: update
            if top_chunk:
                relevance = int(self.get_llm_relevance(top_chunk, query_str))
                keywords = self.get_gpt_keywords(query_str, curr_passage=chunk)
                self.update_weight(keywords, keywords_map, relevance)
            
            query_emb = self.update_embedding(query_emb, keywords_map=keywords_map)


        query_results = dict()
        for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
            avg_score, top_chunks = self.retrieval_for_dest(query=query_emb, dest_chunks=dests_chunks[dest_name], dest_emb=dest_emb)
            query_results[dest_name] = (avg_score, top_chunks)

        sorted_query_results = dict(sorted(query_results.items(), key=lambda item: item[1][0], reverse=True))
        return sorted_query_results 
    

    def sort_prf_chunks(self, query_emb: np.ndarray, chunk_embs: np.ndarray, dest_chunks: list[str]) -> list[str]:
        # embed query reformulation
        score = cosine_similarity(chunk_embs, query_emb).flatten()  # shape [chunk_size]

        # sort chunks
        sort_idx = np.argsort(score)
        chunks = np.array(dest_chunks)  # [chunk_size]
        sort_chunks = chunks[sort_idx].tolist()

        return sort_chunks


    def get_llm_relevance(self, curr_passage: str, original_aspect: str) -> bool:
        prompt = """
        Given a user's travel cities recommendation query, determine whether the provided passage from a city's introduction serve as good evidence for the query.
        
        Answer only in "yes" or "no".
        passage: {curr_passage}
        query: {original_aspect}
        Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        """
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(curr_passage=curr_passage, original_aspect=original_aspect)},
        ]
        response = self.llm.generate(message)
        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        response = json.loads(response)["answer"]

        if 'yes' in response.lower():
            return True
        return False
    

    def get_gpt_keywords(self, aspect_text: str, curr_passage: str) -> list[str]:
        prompt = """
        Given a initial query and a related passage, identify a list of keywords from the passage that once added to the initial query, could be used to better retrieve passages that contain the actual answer. 
        Format your answer in the following JSON format: {{"answer": [KEY WORD LIST]}}, with key being "answer" and value being a python list.
        query: {query}
        passage: {passage}
        keywords:
        """
        prompt = prompt.format(query=aspect_text, passage=curr_passage)
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt},
        ]
        response = self.llm.generate(message)

        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        keywords = json.loads(response)["answer"]

        return keywords


    def update_weight(self, keywords_list: list[str], keywords_map: dict[str, list[int]], relevance: int):
        for key in keywords_list:
            key = key.lower().strip()
            if key not in keywords_map:
                keywords_map[key] = []
                if relevance == 1:
                    keywords_map[key].append([1, relevance])
                else:
                    keywords_map[key].append([0, relevance])
            else:
                if relevance == 1:
                    keywords_map[key][0][0] += self.beta
                    keywords_map[key][0][1] = relevance
                else:
                    keywords_map[key][0][0] -= self.gemma
                    keywords_map[key][0][1] = keywords_map[key][0][1] or relevance
    

    def update_embedding(self, query_emb: np.ndarray, keywords_map: dict[str, list[int]]) -> np.ndarray:
        for key in keywords_map:
            weight = int(keywords_map[key][0][0]) # weight
            query_emb += weight * self.model.encode(key) 

        return query_emb




    


