import abc, os, pickle
import torch
from tqdm import tqdm
from nltk import sent_tokenize

SPLIT_TYPE = {"sentence", "section"}

class LMEmbedder(abc.ABC):
    """
    Abstract class for Language Model Embedder (LM) Embedder
    """
    def __init__(self, model_name: str, split_type: str = "section"):
        self.model_name = model_name
        # check the split type
        if split_type not in SPLIT_TYPE:
            raise ValueError(f"Invalid split_type: {split_type}. Valid options are {SPLIT_TYPE}")
        self.split_type = split_type

    @abc.abstractmethod
    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode the text into embeddings
        :param text:
        """
        pass

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

                chunk_prefix = os.path.join(output_dir, "chunks", self.split_type) # path for chunks
                output_prefix = os.path.join(output_dir, self.model_name.replace('/', '_').lower(), self.split_type) # path for embed vectors
                
                os.makedirs(chunk_prefix, exist_ok=True)
                os.makedirs(output_prefix, exist_ok=True) # create the output dir if neccesary

                chunk_prefix = os.path.join(chunk_prefix, file_prefix)
                output_prefix = os.path.join(output_prefix, file_prefix)

                emb_path = f"{output_prefix}_emb.pkl"
                chunks_path = f"{chunk_prefix}_chunks.pkl"

                if os.path.exists(chunks_path): # if only chunks exist
                    with open(chunks_path, "rb") as f:
                        truncated_chunks = pickle.load(f)
                else: # if chunks do not exist  
                    file_path = os.path.join(data_path, file) # process
                    with open(file_path, "r", encoding='utf-8') as f:
                        text = f.read()
                    chunks = self.split_chunk(text) # split chunks
                    truncated_chunks = [chunk[:18000] if len(chunk) > 18000 else chunk for chunk in chunks] # only keep 18000 tokens
                    
                    with open(chunks_path, "wb") as chunks_file: # save
                        pickle.dump(truncated_chunks, chunks_file)

                if os.path.exists(emb_path): # skip processing if both embeddings and chunks have already been saved
                    continue
                    
                # embed and save chunks for each dest file
                try:
                    embeds = self.encode(truncated_chunks)
                    
                    with open(emb_path, "wb") as emb_file:
                        pickle.dump(embeds, emb_file)

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
        
