import argparse
import numpy as np
from openai import OpenAI
from src.Embedder.LMEmbedder import LMEmbedder
from config import API_KEY


class GPTEmbedder(LMEmbedder):
    """
    Embedder using the GPT-3 model
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = "API_KEY", split_type: str = "section"):
        super().__init__(model_name=model_name, split_type=split_type)
        self.client = OpenAI(api_key=api_key)

    def encode(self, text: str | list[str]) -> np.ndarray:
        """
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text] if isinstance(text, str) else text,
        )

        return np.array([s.embedding for s in response.data])

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using a GPT model.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the directory containing text files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where embeddings should be saved.")
    parser.add_argument("--split_type", type=str, default="section", choices=["sentence", "section"],
                        help="The type of text splitting to apply before embedding.")
    parser.add_argument("-m", "--model_name", type=str, default="text-embedding-3-small",
                        help="The name of the GPT model to use for generating embeddings.")

    args = parser.parse_args()

    embedder = GPTEmbedder(api_key=API_KEY, model_name=args.model_name, split_type=args.split_type)
    embedder.create_embeddings(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()
    
    

