import argparse
import torch
from sentence_transformers import SentenceTransformer
from src.Embedder.LMEmbedder import LMEmbedder

class STEmbedder(LMEmbedder):
    """
    Embedder using sentence transformers
    """
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2", split_type: str = "section"):
        super().__init__(model_name=model_name, split_type=split_type)
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode the text into embeddings
        :param text:
        :return:
        """
        return self.model.encode([text] if isinstance(text, str) else text)
    
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using Sentence Transformers.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the directory containing text files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where embeddings should be saved.")
    parser.add_argument("--split_type", type=str, default="section", choices=["sentence", "section"],
                        help="The type of text splitting to apply before embedding.")
    parser.add_argument("-m", "--model_name", type=str, default="paraphrase-MiniLM-L6-v2",
                        help="The name of the sentence transformer model to use for generating embeddings.")

    args = parser.parse_args()

    embedder = STEmbedder(model_name=args.model_name, split_type=args.split_type)
    embedder.create_embeddings(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()
    

    