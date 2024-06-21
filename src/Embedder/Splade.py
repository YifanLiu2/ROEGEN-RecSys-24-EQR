import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.Embedder.LMEmbedder import LMEmbedder

class Splade_doc(LMEmbedder):
    """
    Splade Embedder for document level
    """

    def __init__(self, model_name: str = "naver/efficient-splade-V-large-doc", split_type: str = "section"):
        super().__init__(model_name=model_name, split_type=split_type)
        self.tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-V-large-doc")
        self.model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-V-large-doc")
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode the text into embeddings
        :param text:
        """
        if isinstance(text, str):
            text = [text]
        all_vecs = []
        batch_size = 32
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i+batch_size]
            tokens = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                tokens = {k: v.to('cuda') for k, v in tokens.items()}
            output = self.model(**tokens)
            doc_vecs = torch.max(
                torch.log(1 + torch.relu(output.logits)) * tokens['attention_mask'].unsqueeze(-1), dim=1
            )[0].detach()  # Remove squeeze() here

            # Ensure doc_vecs is always 2D (batch_size, num_features)
            if doc_vecs.dim() == 1:
                doc_vecs = doc_vecs.unsqueeze(0)

            all_vecs.append(doc_vecs.cpu())  # Move tensors back to CPU if needed
        return torch.cat(all_vecs, dim=0)
    
class Splade_query(LMEmbedder):
    """
    Splade Embedder for query level
    """

    def _init_(self, model_name = "naver/efficient-splade-V-large-query", split_type: str = "section"):
        super()._init_(model_name=model_name, split_type=split_type)
        self.tokenizer = AutoTokenizer.from_pretrained("naver/efficient-splade-V-large-query")
        self.model = AutoModelForMaskedLM.from_pretrained("naver/efficient-splade-V-large-query")
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')


    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Encode the text into embeddings
        :param text:
        """
        if isinstance(text, str):
            text = [text]
        all_vecs = []
        batch_size = 32
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i+batch_size]
            tokens = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            if torch.cuda.is_available():
                tokens = {k: v.to('cuda') for k, v in tokens.items()}
            output = self.model(**tokens)
            doc_vecs = torch.max(
                torch.log(1 + torch.relu(output.logits)) * tokens['attention_mask'].unsqueeze(-1), dim=1
            )[0].detach()  # Remove squeeze() here

            # Ensure doc_vecs is always 2D (batch_size, num_features)
            if doc_vecs.dim() == 1:
                doc_vecs = doc_vecs.unsqueeze(0)

            all_vecs.append(doc_vecs.cpu())  # Move tensors back to CPU if needed
        return torch.cat(all_vecs, dim=0)