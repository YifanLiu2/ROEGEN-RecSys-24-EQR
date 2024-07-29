from src.Retriever.denseRetriever import *
from src.Retriever.sparseRetriever import *
from src.Retriever.proQERetriever import *

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
    chunks_dir = args.chunks_dir
    output_dir = args.output_dir
    num_chunks = (args.num_chunks_broad, args.num_chunks_activity)
    power = args.power

    if num_chunks[0] <= 0 or num_chunks[1] <= 0:
        raise ValueError(f"Invalid number of chunks, should be positive integer: {num_chunks}")
    
    if power <= 0:
        raise ValueError(f"Invalid power, should be positive integer: {power}")
    
    os.makedirs(output_dir, exist_ok=True)

    if embedding_dir is not None:
        if not os.path.exists(embedding_dir):
            raise ValueError(f"Invalid directory path for destination embeddings: {embedding_dir}")
    
    if chunks_dir is not None:
        if not os.path.exists(chunks_dir):
            raise ValueError(f"Invalid directory path for destination chunks: {chunks_dir}")
    
    if args.retrieve_type == "dense":
        retriever = DenseRetriever(model=model, query_path=query_path, chunks_dir=chunks_dir, embedding_dir=embedding_dir, output_dir=output_dir, num_chunks=num_chunks, power=power)
        retriever.run_retrieval()
    elif args.retrieve_type == "BM25":
        retriever = BM25Retriever(query_path=query_path, chunks_dir=chunks_dir,output_dir=output_dir, num_chunks=num_chunks, power=power)
        retriever.run_retrieval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the dense retriever.")
    parser.add_argument("--query_path", required=True, default="output/processed_query_elaborate.pkl", help="Path to the input file containing queries")
    parser.add_argument("--chunks_dir", required=True,help="Directory containing destination chunks")
    parser.add_argument("--embedding_dir", help="Directory containing embeddings")
    parser.add_argument("--retrieve_type", default="dense", help="Type of retrieval to perform")
    parser.add_argument("--emb_type", type=str, choices=TYPE, default="gpt", help="Specify the type of the embedder. Available types are: {}".format(", ".join(sorted(TYPE))))
    parser.add_argument("--output_dir", help="Directory to store retrieval results")
    parser.add_argument("-nb", "--num_chunks_broad", type=int, default=10, help="")
    parser.add_argument("-na", "--num_chunks_activity", type=int, default=3, help="")
    parser.add_argument("--power", type=int, default=5, help="")
    args = parser.parse_args()
    main(args=args)


