import os
# set the working directory to the root of the project
from src.Retriever.retriever import Retriever

from src.Embedder.LMEmbedder import LMEmbedder
from src.Embedder.GPTEmbedder import GPTEmbedder
from src.Embedder.STEmbedder import STEmbedder

import numpy as np
import os

gptembedder = GPTEmbedder(api_key="sk-proj-4dsHn9YkAg7UGKWi6QytT3BlbkFJLapmZydfPkYW4ilePUOV")
stembedder = STEmbedder()
# TESTS
retriever = Retriever(gptembedder)

retriever.run_dense_retrieval(query_path="data/queries.txt", embs_dir="embeddings/text-embedding-3-small/section", percentile=50, output_dir="output")