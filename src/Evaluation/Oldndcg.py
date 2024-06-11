from BaseEvaluator import Evaluator
from fuzzywuzzy import fuzz
from sklearn.metrics import ndcg_score
import numpy as np

class NDCG(Evaluator):

    def __init__(self, k: int, relevance, dr_results, query_mapper=None):
        super().__init__(k, relevance, dr_results, query_mapper)

    def ndcg_at_k(self):
        # get per query and overall and return both
        ndcg_per_query = {}
        ndcg_overall = 0
        total_queries = len(self.relevance)
        for query in self.relevance:
            query_dr = query
            if self.query_mapper:
                query_dr = self.query_mapper[query]

            rel_dests = [(d[0].lower(), d[1]) for d in self.relevance[query]]
            dr_dests = [d[0].lower() for d in self.dr_results[query_dr][:self.k]]

            rel_scores = []
            dr_scores = [d[1][1] for d in self.dr_results[query_dr][:self.k]]


            for d in dr_dests:
                found = False
                for d2, d2_score in rel_dests:
                    if fuzz.ratio(d, d2) > 85:
                        if d != d2:
                            print(f"Matched {d} to {d2} with {fuzz.ratio(d, d2)}% similarity.")
                        rel_scores.append(d2_score)
                        found = True
                        break
                if not found:
                    rel_scores.append(0)
            if len(rel_scores) > 1:
                ndcg_per_query[query] = ndcg_score(np.array([rel_scores]), np.array([dr_scores]))
            else:
                ndcg_per_query[query] = 0
            ndcg_overall += ndcg_per_query[query]
        
        ndcg_overall /= total_queries
        return ndcg_per_query, ndcg_overall