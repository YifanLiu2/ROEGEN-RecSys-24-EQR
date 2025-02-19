from config import API_KEY
import argparse
from .GPTEmbedder import GPTEmbedder
from .STEmbedder import STEmbedder

EMBEDDER_TYPES = {"gpt", "st"}

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using either a GPT model or a sentence transformer.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the directory containing the input text files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where the generated embeddings should be saved.")
    parser.add_argument("--split_type", type=str, default="section", choices=["sentence", "section"],
                        help="Method for splitting text before embedding. Available options: 'sentence' or 'section'.")
    parser.add_argument("--emb_type", type=str, choices=EMBEDDER_TYPES, default="gpt", 
                        help="Type of embedder to use. Available options: {}".format(", ".join(sorted(EMBEDDER_TYPES))))
    parser.add_argument("--emb_name", type=str, help="Name of the sentence transfomer embedder to use.")
    args = parser.parse_args()

    if args.emb_type == "gpt":
        embedder = GPTEmbedder(api_key=API_KEY, split_type=args.split_type)
        embedder.create_embeddings(args.data_path, args.output_dir)
        
    elif args.emb_type == "st":
        if not args.emb_name:
            raise ValueError("Please provide the name of the sentence transformer embedder to use.")
        embedder = STEmbedder(split_type=args.split_type, model_name=args.emb_name)
        embedder.create_embeddings(args.data_path, args.output_dir)

    else:
        raise ValueError("Invalid embedder type. Available types are: {}".format(", ".join(sorted(EMBEDDER_TYPES))))

if __name__ == "__main__":
    main()
