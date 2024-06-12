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
stembedder = STEmbedder()

# TESTS
# retriever = DenseRetriever(model=gptembedder, query_path="output/processed_query_elaborate.pkl", embedding_dir="embeddings/text-embedding-3-small/section", percentile=97, output_path="output/dense_results_total_ela_top3.json")
retriever = DenseRetriever(model=stembedder, query_path="output/processed_query_elaborate.pkl", embedding_dir="embeddings\paraphrase-minilm-l6-v2\section", percentile=97, output_path="output/dense_results_total_ela_top3_ST.json")
retriever.run_retrieval()

