import json, abc
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from src.LLM.GPTChatCompletion import LLM, GPTChatCompletion
from src.Retriever.abstractRetriever import *
from src.Retriever.sparseRetriever import *
from src.Retriever.denseRetriever import *
from src.Embedder.LMEmbedder import LMEmbedder
from src.LLM.GPTChatCompletion import *
from src.Retriever.sparseRetriever import GPTChatCompletion
from config import API_KEY

class ProQERetriever(AbstractRetriever):
    """
    """
    def __init__(self, query_path: str, output_path: str, chunks_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), model: LMEmbedder = None, num_chunks: Optional[int] = None, percentile: Optional[float] = None, threshold: Optional[float] = None, num_iter: int = 5, beta: int = 1, gemma: int = 0):
        super().__init__(query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, model=model, num_chunks=num_chunks, percentile=percentile, threshold=threshold)
        self.llm = llm
        self.num_iter = num_iter
        self.beta = beta
        self.gemma = gemma    

    def retrieval_for_query(self, query: Query | str, dests_chunks: dict[str, list[str]], dests_embs: dict[str, np.ndarray] = None) -> dict[str, tuple[float, dict[str, list[str]]]]: 
        """
        Retrieval for a single query for ProQeRetriever.
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        # return format: {dest: (dest_score, {"aspect": top_chunk})}
        final_results = dict()
        query = query

        visited = set()

        # fetch the chunking method
        if self.threshold:
            chunk_method = self.threshold_method

        elif self.num_chunks:
            chunk_method = self.top_chunk_method
        
        elif self.percentile:
            chunk_method = self.percentile_method

        # iterative PRF over num_iter
        for i in range(self.num_iter): 
            if i == self.num_iter - 1:
                num_chunks = 1
            # retrieve the pseudo document for all apsects 
            query_results = dict()

            if isinstance(query, str):
                aspects = [Aspect(query)]
            else:
                aspects = query.get_all_aspects()

            if self.cls_type == "sparse":
                for dest_name, dest_chunks in tqdm(dests_chunks.items(), desc="Processing destinations"):
                    dest_result = self.retrieval_for_dest(aspects=aspects, dest_chunks=dest_chunks, chunk_method=chunk_method)
                    query_results[dest_name] = dest_result
                
            elif self.cls_type == "dense":
                for dest_name, dest_emb in tqdm(dests_embs.items(), desc="Processing destinations"):
                    dest_result = self.retrieval_for_dest(aspects=aspects, dest_chunks=dests_chunks[dest_name], chunk_method=chunk_method, dest_emb=dest_emb)
                    query_results[dest_name] = dest_result

            # rank the score for each aspect
            ranked_results = dict()
            for dest, dest_result in query_results.items():
                for aspect, (score, top_chunk) in dest_result.items():
                    if aspect not in ranked_results:
                        ranked_results[aspect] = []
                    ranked_results[aspect].append((dest, score, top_chunk))

            for aspect in ranked_results:
                ranked_results[aspect].sort(key=lambda x: x[1], reverse=True)
            
            keywords_map = dict()

            # retrieve the pseudo document for one aspect
            for aspect in aspects:
                aspect_text = aspect.get_new_description()

                # find the top 1 chunks for the aspect
                top_dest = None
                for i, dest in enumerate(ranked_results[aspect_text]):
                    if dest[0] not in visited:
                        top_dest = i
                        visited.add(dest[0])
                        break

                # check relevance
                relevance = 0
                is_relevant = self.get_llm_relevance(ranked_results[aspect_text][top_dest][2], aspect_text)
                keywords = self.get_gpt_keywords(aspect_text, curr_passage=ranked_results[aspect_text][top_dest][2])

                if is_relevant:
                    # print(query_results[top_dest][1][aspect_text][0])
                    relevance = 1

                # update weight and aspect
                self.update_weight(keywords, keywords_map, relevance)
                new_aspect_text = self.update_aspect(aspect_text=aspect_text, keywords_map=keywords_map)
                aspect.set_new_description(new_aspect_text)

        for dest_name, dest_result in query_results.items(): # for the last query results
            dest_result = self.avg_fusion(dest_result)
            final_results[dest_name] = dest_result
            
        return final_results


    @abc.abstractmethod
    def update_aspect(self, aspect_text: str, keywords_map: dict[str, list[int]]) -> str:
        pass


    def get_llm_relevance(self, curr_passage: str, original_aspect: str) -> bool:
        prompt = f"""Is the following passage related to the query?
        passage: {curr_passage}
        query: {original_aspect}
        Answer only in "yes" or "no".
        """
        message = [
            {"role": "user", "content": prompt},
        ]
        response = self.llm.generate(message)

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


class SparseProQERetriever(ProQERetriever):
    """
    """
    cls_type = "sparse"
    
    def __init__(self, query_path: str, output_path: str, chunks_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), model: LMEmbedder = None, num_iter: int = 5, beta: int = 1, gemma: int = 0, num_chunks: Optional[int] = None, percentile: Optional[float] = None, threshold: Optional[float] = None):
        super().__init__(query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, llm=llm, model=model, num_chunks=num_chunks, percentile=percentile, threshold=threshold, num_iter=num_iter, beta=beta, gemma=gemma)

    def update_aspect(self, aspect_text: str, keywords_map: dict[str, list[int]]) -> str:
        for key in keywords_map:
            times = int(keywords_map[key][0][0]) # weight
            for _ in range(times): # how many times to append that keyword to the query
                aspect_text += " " + key
        
        return aspect_text
    
    def retrieval_for_dest(self, aspects: list[Aspect], dest_chunks: list[str], chunk_method: Callable) -> dict[str, tuple[float, list[str]]]:
        """
        Perform sparse retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        # return format: {"aspect": (score, top_chunk)}
        dest_result = dict()
        corpus = dest_chunks
        tokenized_corpus = [doc.split(" ") for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)
        for a in aspects:
            a_text = a.get_new_description()
            # compute bm25 scores for a_text and dest_chunks
            tokenized_a = word_tokenize(a_text)

            # get the score for the top chunks
            score = bm25.get_scores(tokenized_a)
            top_idx = chunk_method(score)
            top_score = score[top_idx]
            avg_score = np.sum(top_score) / top_score.shape[0]

            # retrieve top chunks
            chunks = np.array(dest_chunks) # [chunk_size]
            top_chunks = chunks[top_idx].tolist()

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result

class DenseProQERetriever(ProQERetriever):
    """
    Dense retriever for ProQERetriever.
    """
    cls_type = "dense"

    def __init__(self, model: LMEmbedder, query_path: str, output_path: str, embedding_dir: str, chunks_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), num_iter: int = 5, beta: int = 1, gemma: int = 0, num_chunks: Optional[int] = None, percentile: Optional[float] = None, threshold: Optional[float] = None):
        super().__init__(query_path=query_path, output_path=output_path, chunks_dir=chunks_dir, llm=llm, model=model, num_iter=num_iter, beta=beta, gemma=gemma, num_chunks=num_chunks, percentile=percentile, threshold=threshold)
        self.dense_embedding_dir = embedding_dir
        self.model = model
    
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
            description_emb = self.model.encode(a_text) # shape [1, emb_size]
            score = cosine_similarity(dest_emb, description_emb).flatten() # shape [chunk_size]

            # extract top 3 idx and top 3 score
            top_idx = chunk_method(score)
            top_score = score[top_idx]
            
            # Check if top_score is not empty and does not contain NaN values
            if top_score.size > 0 and not np.isnan(top_score).any():
                avg_score = np.sum(top_score) / top_score.shape[0]  # a scalar score
            else:
                avg_score = 0  # or some default value, depending on your use case


            # retrieve top chunks
            chunks = np.array(dest_chunks) # [chunk_size]
            top_chunks = chunks[top_idx].tolist()

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result

    def update_aspect(self, aspect_text: str, keywords_map: dict[str, list[int]]) -> str:
        for key in keywords_map:
            times = int(keywords_map[key][0][0]) # weight
            for _ in range(times): # how many times to append that keyword to the query
                aspect_text += " " + key
        
        return aspect_text


    


