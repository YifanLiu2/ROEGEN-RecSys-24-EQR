import abc, os, pickle
import torch
from tqdm import tqdm
from nltk import sent_tokenize

SPLIT_TYPE = {"sentence", "section"}

class LMEmbedder(abc.ABC):
    """
    Abstract base class for a Embedder.
    """

    def __init__(self, model_name: str, split_type: str = "section", concate_city_name: bool = False):
        self.model_name = model_name
        if split_type not in SPLIT_TYPE:
            raise ValueError(f"Invalid split_type: {split_type}. Valid options are {SPLIT_TYPE}")
        self.split_type = split_type
        self.concate = concate_city_name

    @abc.abstractmethod
    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode the given text into embeddings.

        text (str or list[str]): Text or list of text segments to be encoded.
        """
        pass

    def create_embeddings(self, data_path: str, output_dir: str):
        """
        Create embeddings for all text files.

        data_path (str): Path to the directory containing input text files.
        output_dir (str): Path to the directory where embeddings should be saved.
        """
        for file in tqdm(os.listdir(data_path)):
            file_prefix = os.path.splitext(file)[0]

            chunk_prefix = os.path.join(output_dir, "chunks", self.split_type)
            output_prefix = os.path.join(output_dir, self.split_type)
            
            os.makedirs(chunk_prefix, exist_ok=True)
            os.makedirs(output_prefix, exist_ok=True)

            chunk_prefix = os.path.join(chunk_prefix, file_prefix)
            output_prefix = os.path.join(output_prefix, file_prefix)

            emb_path = f"{output_prefix}_emb.pkl"
            chunks_path = f"{chunk_prefix}_chunks.pkl"

            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    truncated_chunks = pickle.load(f)
            else:
                file_path = os.path.join(data_path, file)
                with open(file_path, "r", encoding='utf-8', errors="ignore") as f:
                    text = f.read()
                chunks = self.split_chunk(text)
                if self.concate:
                    print("Concatenating city name")
                    chunks = [f"{file_prefix}: {chunk}" for chunk in chunks]
                truncated_chunks = [chunk[:18000] if len(chunk) > 18000 else chunk for chunk in chunks]
                
                with open(chunks_path, "wb") as chunks_file:
                    pickle.dump(truncated_chunks, chunks_file)

            if os.path.exists(emb_path):
                continue
                
            try:
                embeds = self.encode(truncated_chunks)
                
                with open(emb_path, "wb") as emb_file:
                    pickle.dump(embeds, emb_file)

            except Exception as e:
                print(f"Failed to embed {file_prefix}: {e}")

    def split_chunk(self, doc: str) -> list[str]:
        """
        Split a document into chunks based on the specified split type.

        doc (str): A string containing the document to split.
        """
        if self.split_type == "sentence":
            return sent_tokenize(doc)
        elif self.split_type == "section":
            return [section.strip() for section in doc.split('\n') if section.strip()]