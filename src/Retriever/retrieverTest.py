import os
# set the working directory to the root of the project
from src.Retriever.denseRetriever import *

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *

import numpy as np
import os
from config import API_KEY

gptembedder = GPTEmbedder(api_key=API_KEY)
stembedder = STEmbedder()
# TESTS
# retriever = DenseRetriever(model=gptembedder, query_path="data/queries.txt", embedding_dir="embeddings/text-embedding-3-small/section", percentile=90, output_path="output/dense_results.json")
# retriever.run_dense_retrieval()

retriever_qe = DenseRetriever(model=gptembedder, query_path="data/queries.txt", embedding_dir="embeddings/text-embedding-3-small/section", percentile=90, output_path="output/bruh.json")
retriever_qe.run_dense_retrieval()