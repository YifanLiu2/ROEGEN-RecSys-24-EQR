from config import API_KEY
import argparse
from .GPTEmbedder import *
from .STEmbedder import *

TYPE = {"gpt", "st"}

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using a GPT model.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the directory containing text files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where embeddings should be saved.")
    parser.add_argument("--split_type", type=str, default="section", choices=["sentence", "section"],
                        help="The type of text splitting to apply before embedding.")
    parser.add_argument("--emb_type", type=str, choices=TYPE, default="gpt", help="Specify the type of the embedder. Available types are: {}".format(", ".join(sorted(TYPE))))
    args = parser.parse_args()

    if args.emb_type == "gpt":
        embedder = GPTEmbedder(api_key=API_KEY, split_type=args.split_type)
        embedder.create_embeddings(args.data_path, args.output_dir)
        
    elif args.emb_type == "st":
        embedder = STEmbedder(split_type=args.split_type)
        embedder.create_embeddings(args.data_path, args.output_dir)

    else:
        raise ValueError("Invalid embedder type. Available types are: {}".format(", ".join(sorted(TYPE))))

if __name__ == "__main__":
    main()