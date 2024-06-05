import pandas as pd
from DataParser import DataParser

class RelevanceParser(DataParser):

    def __init__(self):
        self.path = "queries_valid_destinations_relevance.csv"
    
    def parse_dataset(self):
        with open(self.path, "r") as f:
            df = pd.read_csv(self.path, encoding='utf-8-sig')
            queries = sorted(list(set(df["Query"].values)))
            
            relevance = {}
            for i, row in df.iterrows():
                query = row["Query"]
                destination = row["City"]
                rel_score = row["Relevance"]
                if query not in relevance:
                    relevance[query] = []
                relevance[query].append((destination, rel_score))
            
            for query in relevance:
                relevance[query] = sorted(relevance[query], key=lambda x: x[1], reverse=True)

            return queries, relevance