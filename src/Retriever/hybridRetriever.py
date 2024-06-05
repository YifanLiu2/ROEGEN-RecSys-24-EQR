from numpy import ndarray
from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *
from src.Retriever.denseRetriever import *
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridRetriever(AbstractRetriever):
    def __init__(self, model: LMEmbedder, query_path: str, dense_embedding_dir: str, output_path: str, percentile: float = 10):
        super().__init__(model, query_path, dense_embedding_dir, output_path, percentile)

    def load_dest_embeddings(self) -> tuple[dict[str, list[str]], dict[str, np.ndarray], dict[str, np.ndarray] | None]:
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
    
    def retrieval(self, queries: list[Query], dests_emb: dict[str, ndarray], dests_chunks: dict[str, list[str]], percentile: float) -> dict:
        """
        Perform dense retrieval using the model and the queries
        :param queries: list of queries
        :param dests_emb: dict of embeddings for each dest file
        :param dests_chunks: dict of chunks for each dest file
        :param percentile: percentile to use for the threshold
        :return: dict of the retrieved documents for each query
        """
        dense_results = dict()
        final_results = dict()
        # for each query
        for q in queries:
            print("-----------------------------")
            print(f"Process query: {q.description}")
            dense_results[q.description] = dict()
            descriptions = q.get_descriptions()
            weights = q.get_description_weights()
            assert len(descriptions) == len(weights) # one to one map between description and weight            
            # Dense retrieval
            # for each destination

            for dest_name, dest_emb in tqdm(dests_emb.items(), desc="Processing destinations using dense retrieval"): # shape [chunk_size, emb_size]
                # assume aggregate method is sum
                dest_score = 0
                sub_dense_results = dict()  # results for subquery
                # for each subquery
                for d, w in zip(descriptions, weights):
                    description_emb = self.model.encode(d) # shape [1, emb_size]
                    score = cosine_similarity(dest_emb, description_emb).flatten() # shape [chunk_size]
                    threshold = np.percentile(score, percentile) # determine the threshold
                    
                    # extract top idx and top score
                    top_idx = np.where(score >= threshold)[0]
                    top_score = score[score >= threshold]
                    avg_score = np.sum(top_score) / top_score.shape[0] # a scalar score

                    # retrieve top chunks
                    chunks = np.array(dests_chunks[dest_name]) # [chunk_size]
                    top_chunks = chunks[top_idx].tolist()
        
                    # store subquery results
                    sub_dense_results[d] = top_chunks
                    dest_score += avg_score * w # avg score weighted by w
            
                # store dest results
                dense_results[q.description][dest_name] = (dest_score, sub_dense_results)


            final_results[q.description] = dict()
            # Sparse retrieval
            specific_constraint = q.get_specific_constraints()
            # Compute sparse retrieval for each destination on SpecificConstraint
            sparse_Embedder = TfidfVectorizer()
            # Embed the specific constraint and all the destinations chunks
            if specific_constraint:
                # Use sparse embeddings to compute the score
                for dest_name, dest_chunks in tqdm(dests_chunks.items(), desc="Processing destinations using sparse retrieval"):
                    # Combine the specific constraint and the destination chunks
                    all_chunks = specific_constraint + dest_chunks
                    all_emb = sparse_Embedder.fit_transform(all_chunks).toarray()
                    specific_constraint_emb = all_emb[0].reshape(1, -1)
                    dest_sparse_emb = all_emb[1:]
                    score = cosine_similarity(dest_sparse_emb, specific_constraint_emb).flatten()
                    threshold = np.percentile(score, percentile)
                    top_idx = np.where(score >= threshold)[0]
                    top_score = score[score >= threshold]
                    avg_score = np.sum(top_score) / top_score.shape[0]
                    chunks = np.array(dest_chunks)
                    top_chunks = chunks[top_idx].tolist()
        
                    # compute sparse retrieval results
                    sparse_score = avg_score * 0.5
                    d_score = dense_results[q.description][dest_name][0]
                    d_chunks = dense_results[q.description][dest_name][1]
                    final_results[q.description][dest_name] = (d_score + sparse_score, d_chunks, top_chunks)
                    
            else:
                final_results[q.description] = dense_results[q.description]

        return final_results
        

    