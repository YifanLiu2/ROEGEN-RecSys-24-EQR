import os
# set the working directory to the root of the project
from src.Retriever.denseRetriever import *
from src.Retriever.hybridRetriever import *

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *

import numpy as np
import os
from config import API_KEY

gptembedder = GPTEmbedder(api_key=API_KEY)
# stembedder = STEmbedder(model_name="sentence-transformers/msmarco-distilbert-base-tas-b")
# stembedder = STEmbedder()
# TESTS

# test dense retriever
retriever = DenseRetriever(model=gptembedder, query_path="data/queries.txt", embedding_dir="embeddings/text-embedding-3-small/section", percentile=97, output_path="output/dense_results_total_ela_top3.json")
retriever.run_retrieval()

# test dense retriever with query expansion
# retriever_qe = DenseRetrieverQE(model=gptembedder, query_path="output/processed_query.pkl", embedding_dir="embeddings/text-embedding-3-small/section", percentile=97, output_path="output/dense_results_with_qe_v2_top3_gpt.json")

# test hybrid retriever with query expansion
# retriever_qe = HybridRetriever(model=stembedder, query_path="output/processed_query.pkl", dense_embedding_dir="embeddings/paraphrase-minilm-l6-v2/section", percentile=97, output_path="output/hybrid_results_with_qe_v2_top3.json")
# retriever_qe.run_retrieval()