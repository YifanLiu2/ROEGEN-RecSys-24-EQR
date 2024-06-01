import os
import pickle

import numpy as np
from nltk import sent_tokenize
from openai import OpenAI
from tqdm import tqdm

from src.Embedder.LLMEmbedder import LLMEmbedder


class GPTEmbedder(LLMEmbedder):
    """
    Embedder using the GPT-3 model
    """
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str = "API_KEY"):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode the text into embeddings
        :param text:
        :return:
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=[text] if isinstance(text, str) else text,
        )

        return np.array([s.embedding for s in response.data])

    def create_embeddings(self, data_path: str, index_dir: str):
        """
        Create embeddings for all files in the data_path and save them in the index_dir
        :param data_path:
        :param index_dir:
        """
        # go through all files in the data_path
        for file in tqdm(os.listdir(data_path)):
            if file.endswith(".txt"):

                file_prefix = file.split(".")[0]
                index_name = f"{index_dir}/{file_prefix}"

                if os.path.exists(f"{index_name}_emb.pkl") and os.path.exists(f"{index_name}_sentences.pkl"):
                    continue

                with open(os.path.join(data_path, file), "r", encoding='utf-8') as f:
                    text = f.read()
                    sentences = sent_tokenize(text)

                    for i in range(len(sentences)):
                        sent = sentences[i]
                        if len(sent) > 18000:
                            sentences[i] = sent[:18000]

                    try:
                        embeds = self.encode(sentences)
                        if embeds is None:
                            print(f"Failed to embed {file_prefix}")
                            continue
                        with open(f"{index_name}_emb.pkl", "wb") as f:
                            pickle.dump(embeds, f)
                        with open(f"{index_name}_sentences.pkl", "wb") as f:
                            pickle.dump(sentences, f)

                        del embeds
                    except Exception as e:
                        print(f"Failed to embed {file_prefix}: {e}")
                    del sentences
