from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
from nltk import sent_tokenize
from tqdm import tqdm


class TASbEmbedder:
    """
    Embedder using the TAS-B model
    """

    def __init__(self, model_name: str = "sentence-transformers/msmarco-distilbert-base-tas-b"):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode the text into embeddings
        :param text:
        :return:
        """
        return self.model.encode(text)

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
