import torch
from openai import OpenAI
from src.Embedder.LMEmbedder import LMEmbedder

class GPTEmbedder(LMEmbedder):
    """
    Embedder using the GPT-3 model
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = "API_KEY", split_type: str = "section"):
        super().__init__(model_name=model_name, split_type=split_type)
        self.client = OpenAI(api_key=api_key)

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text] if isinstance(text, str) else text,
        )

        return torch.Tensor([s.embedding for s in response.data])

    

