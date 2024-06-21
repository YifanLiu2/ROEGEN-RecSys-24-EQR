from src.Retriever.denseRetriever import *
from src.Retriever.sparseRetriever import *

from src.Embedder.GPTEmbedder import *
from src.Embedder.STEmbedder import *

import argparse, os
from config import API_KEY

TYPE = {"gpt", "st"}

def main(args):
    # Select the model to use
    model_name = args.emb_type
    if model_name == "gpt":
        model = GPTEmbedder(api_key=API_KEY)
    elif model_name == "st":
        model = STEmbedder()
    
    query_path = args.query_path
    embedding_dir = args.embedding_dir
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if embedding_dir is not None:
        if not os.path.exists(embedding_dir):
            raise ValueError(f"Invalid directory path for destination embeddings: {embedding_dir}")
    
    if args.retrieve_type == "dense":
        retriever = DenseRetriever(model=model, query_path=query_path, embedding_dir=embedding_dir, output_path=output_path)
        retriever.run_retrieval()
    elif args.retrieve_type == "BM25":
        retriever = SparseRetriever(query_path=query_path, output_path=output_path)
        retriever.run_retrieval()
    elif args.retrieve_type == "splade":
        retriever = SpladeRetriever(query_path=query_path, output_path=output_path)
        retriever.run_retrieval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the dense retriever.")
    parser.add_argument("-q", "--query_path", required=True, default="output/processed_query_elaborate.pkl", help="Path to the input file containing queries")
    parser.add_argument("-e", "--embedding_dir", help="Directory containing embeddings")
    parser.add_argument("-x", "--retrieve_type", default="dense", help="Type of retrieval to perform")
    parser.add_argument("--emb_type", type=str, choices=TYPE, default="gpt", help="Specify the type of the embedder. Available types are: {}".format(", ".join(sorted(TYPE))))
    parser.add_argument("-o", "--output_path", help="Path to store retrieval results")
    args = parser.parse_args()
    main(args=args)


