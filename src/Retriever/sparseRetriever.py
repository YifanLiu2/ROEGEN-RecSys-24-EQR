from src.Retriever.denseRetriever import *
from rank_bm25 import BM25Okapi
from fastembed import SparseTextEmbedding

class SparseRetriever(AbstractRetriever):
    """
    Concrete DenseRetriever class.
    """
    def __init__(self, query_path: str, output_path: str, percentile: float = 10):
        super().__init__(model=None, query_path=query_path, output_path=output_path, percentile=percentile)
    

    def retrieval_for_dest(self, aspects: list[Aspect], dest_chunks: list[str], percentile: float) -> dict[str, tuple[float, list[str]]]:
        """
        Perform dense retrieval for each query.

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
            tokenized_a = a_text.split(" ")

            top_chunks = bm25.get_top_n(tokenized_a, corpus, n=3)
            # get the score for the top chunks
            scores = bm25.get_scores(tokenized_a)
            scores.sort()

            avg_score = sum(scores[-3:]) / 3

            # store results
            dest_result[a_text] = (avg_score, top_chunks)
        return dest_result
    
class SpladeRetriever(SparseRetriever):
    """
    Concrete SparseRetriever class.
    """

    def __init__(self, query_path: str, output_path: str, percentile: float = 10):
        super().__init__(query_path=query_path, output_path=output_path, percentile=percentile)
        self.model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    
    def retrieval_for_dest(self, aspects: list[Aspect], dest_chunks: list[str], percentile: float) -> dict[str, tuple[float, list[str]]]:
        """
        Perform dense retrieval for each query.

        :param queries: list[str], list of query strings.
        :param dests_chunks: dict[str, list[str]], dictionary of destination names to lists of associated text chunks.
        :param percentile: float, percentile to determine the similarity threshold for filtering results.
        :return: dict[str, dict[str, tuple[float, list[str]]]], structured results with scores and top matching chunks.
        """
        dest_result = dict()
        for a in aspects:
            a_text = a.get_new_description()
            # compute embeddings for a_text and dest_chunks
            total_text = [a_text] + dest_chunks
            embeddings = list(self.model.embed(total_text))
            a_embedding = np.array(embeddings[0].values)
            print(a_embedding.shape)
            
            dest_embeddings = [emb.values for emb in embeddings[1:]]
            print(dest_embeddings[0].shape)
                            

            # compute cosine similarity between a_embedding and dest_embeddings
            score = cosine_similarity(dest_embeddings, a_embedding).flatten() # shape [chunk_size]
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

    