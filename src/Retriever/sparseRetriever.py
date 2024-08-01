from src.Retriever.denseRetriever import *
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from src.Entity.query import AbstractQuery


class SparseRetriever(AbstractRetriever):
    """
    Concrete SparseRetriever class.
    """
    cls_type = "sparse"

class BM25Retriever(SparseRetriever):
    """
    Concrete SparseRetriever class.
    """
    def retrieval_for_dest(self, query: AbstractQuery, dest_chunks: list[str]) -> dict[str, tuple[float, list[str]]]:
        """
        Perform sparse retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        corpus = dest_chunks
        tokenized_corpus = [word_tokenize(doc) for doc in corpus]
        num_chunks = self.num_chunks

        # compute bm25 scores for a_text and dest_chunks      
        reformulation = query.get_reformulation()
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_r = word_tokenize(reformulation)

        # get the score 
        score = bm25.get_scores(tokenized_r)
        top_idx = np.argsort(score)[-num_chunks:]
        top_score = score[top_idx]
        avg_score = self.calculate_city_score(top_score)

        # retrieve top chunks
        chunks = np.array(dest_chunks) # [chunk_size]
        top_chunks = chunks[top_idx].tolist()

        # retrieve top chunks
        chunks = np.array(dest_chunks)  # [chunk_size]
        top_chunks = chunks[top_idx].tolist()

        return avg_score, top_chunks

    
# class SpladeRetriever(SparseRetriever):
#     """
#     Concrete SparseRetriever class.
#     """

#     def __init__(self, query_path: str, chunks_dir: str, output_dir: str, num_chunks: int = 3, batchsize: int = 8):
#         super().__init__(query_path=query_path, output_dir=output_dir, chunks_dir=chunks_dir, num_chunks=num_chunks)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"Using device: {self.device}")
#         self.model = model.Splade(
#             model=AutoModelForMaskedLM.from_pretrained("naver/splade_v2_max").to(self.device),
#             tokenizer=AutoTokenizer.from_pretrained("naver/splade_v2_max"),
#             device=self.device
#         )
#         self.batch_size = batchsize
    
#     def retrieval_for_dest(self, aspects: list[AbstractQuery], dest_chunks: list[str], num_chunks: int = 3) -> dict[str, tuple[float, list[str]]]:
#         """
#         Perform sparse retrieval for each query.

#         :param queries: list[str], list of query strings.
#         :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
#         :param percentile: float, percentile to determine the similarity threshold for filtering results.
#         :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
#         """
#         dest_result = dict()
#         aspects_text = [a.get_new_description() for a in aspects]
#         docs = [{"id": i, "chunks": chunk} for i, chunk in enumerate(dest_chunks)]
#         retriever = retrieve.SpladeRetriever(
#             key="id", # Key identifier of each document.
#             on="chunks", # Fields to search.
#             model=self.model # Splade retriever.
#         )

#         retriever = retriever.add(
#             documents=docs,
#             batch_size=self.batch_size,
#             k_tokens=256, # Number of activated tokens.
#         )

#         splade_results = retriever(
#             aspects_text, # AbstractQuery
#             k_tokens=20, # Maximum number of activated tokens.
#             k = num_chunks, # Number of top-k documents to retrieve.
#             batch_size=self.batch_size
#         )
        
#         count = 0
#         for result in splade_results:
#             top_chunks = []
#             score =[]
#             for r in result:
#                 top = next((doc["chunks"] for doc in docs if doc["id"] == r["id"]), None)
#                 if top is not None:
#                     top_chunks.append(top)
#                     score.append(r["similarity"])
                    
#             avg_score = sum(score) / len(score)
#             dest_result[aspects_text[count]] = (avg_score, top_chunks)
#             count += 1
#         return dest_result

       
