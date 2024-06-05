from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import numpy as np
from tqdm import tqdm
import os
import pickle
from nltk import sent_tokenize
SPLIT_TYPE = {"sentence", "section"}

class SparseEmbedder:
    def __init__(self, split_type: str = "section"):
        self.vectorizer = TfidfVectorizer()
        self.model_name = "sparse-embedder"
        # check the split type
        if split_type not in SPLIT_TYPE:
            raise ValueError(f"Invalid split_type: {split_type}. Valid options are {SPLIT_TYPE}")
        self.split_type = split_type

    def encode(self, text: str | list[str]) -> np.ndarray:
        """
        Encode the text into embeddings
        :param text:
        """
        return self.vectorizer.fit_transform(text).toarray()
    
    def create_embeddings(self, data_path: str, output_dir: str):
        """
        Create embeddings for all files in the data_path and save them in the output_dir
        :param data_path:
        :param output_dir:
        """
        # process all dest file
        for file in tqdm(os.listdir(data_path)):
            if file.endswith(".txt"):
                file_prefix = os.path.splitext(file)[0]
                output_prefix = os.path.join(output_dir, self.model_name.replace('/', '_').lower(), self.split_type)
                os.makedirs(output_prefix, exist_ok=True) # create the output dir if neccesary
                output_prefix = os.path.join(output_prefix, file_prefix)

                emb_path = f"{output_prefix}_emb.pkl"
                chunks_path = f"{output_prefix}_chunks.pkl"

                if os.path.exists(emb_path) and os.path.exists(chunks_path): # skip processing if both embeddings and chunks have already been saved
                    continue

                file_path = os.path.join(data_path, file)
                with open(file_path, "r", encoding='utf-8') as f:
                    text = f.read()
                
                chunks = self.split_chunk(text) # split chunks
                truncated_chunks = [chunk[:18000] if len(chunk) > 18000 else chunk for chunk in chunks] # only keep 18000 tokens

                # embed and save chunks for each dest file
                try:
                    embeds = self.encode(truncated_chunks)
                    
                    with open(emb_path, "wb") as emb_file, open(chunks_path, "wb") as chunks_file:
                        pickle.dump(embeds, emb_file)
                        pickle.dump(truncated_chunks, chunks_file)

                except Exception as e:
                    print(f"Failed to embed {file_prefix}: {e}")

    def split_chunk(self, doc: str) -> list[str]:
        """
        Splits the document into chunks based on split type.

        :param doc: A string containing the document to split.
        :return: A list of chunks extracted from the document.
        """
        if self.split_type == "sentence":
            return sent_tokenize(doc)
        elif self.split_type == "section":
            return [section.strip() for section in doc.split('\n\n') if section.strip()]
        
def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text files using a sparse embedding model.")
    parser.add_argument("-d", "--data_path", type=str, required=True, help="Path to the directory containing text files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path where embeddings should be saved.")
    parser.add_argument("--split_type", type=str, default="section", choices=["sentence", "section"],
                        help="The type of text splitting to apply before embedding.")

    args = parser.parse_args()

    embedder = SparseEmbedder(split_type=args.split_type)
    embedder.create_embeddings(args.data_path, args.output_dir)

if __name__ == "__main__":
    main()

            