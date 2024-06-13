import os
# set the working directory to the root of the project
from src.Retriever.denseRetriever import *
from src.Retriever.hybridRetriever import *

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *

import numpy as np
import os
from config import API_KEY

def main(args):
    # Select the model to use
    modle_name = args.modle_name
    if modle_name == "GPT":
        model = GPTEmbedder(api_key=API_KEY)
    elif modle_name == "ST":
        model = STEmbedder()
    else:
        raise ValueError("Invalid model name, should be either 'GPT' or 'ST'")
    
    query_path = args.query_path
    embedding_dir = args.embedding_dir
    output_dir = args.output_dir

    retriever = DenseRetriever(model=model, query_path=query_path, embedding_dir=embedding_dir, output_path=output_dir)
    retriever.run_retrieval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the dense retriever.")
    parser.add_argument("-q", "--query_path", required=True, default="output/processed_query_elaborate.pkl", help="Path to the input file containing queries")
    parser.add_argument("-e", "--embedding_dir", required=True, help="Directory containing embeddings")
    parser.add_argument("-m", "--modle_name", required=True, help="Embedding model to use for processing queries")
    parser.add_argument("-o", "--output_dir", help="Directory to store processed queries", default="output")
    args = parser.parse_args()
    main(args=args)


