import abc

import numpy as np


class LLMEmbedder(abc.ABC):
    """
    Abstract class for Language Model Embedder (LLM) Embedder
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def encode(self, text: str | list[str]) -> np.ndarray:
        """
        Encode the text into embeddings
        :param text:
        """
        pass
