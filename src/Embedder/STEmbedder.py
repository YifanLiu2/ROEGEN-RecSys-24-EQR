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

    

    