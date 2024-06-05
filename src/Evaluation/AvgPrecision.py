from BaseEvaluator import Evaluator
from sklearn.metrics import average_precision_score
from fuzzywuzzy import fuzz
import numpy as np

class AvgPrecision(Evaluator):

    def __init__(self, k: int, relevance, dr_results, query_mapper=None):
        super().__init__(k, relevance, dr_results, query_mapper)

    def average_precision_at_k(self):
        # get per query and overall and return both
        ap_per_query = {}
        ap_overall = 0
        total_queries = len(self.relevance)
        for query in self.relevance:
            query_dr = query
            if self.query_mapper:
                query_dr = self.query_mapper[query]
                
            rel_dests = [(d[0].lower(), d[1]) for d in self.relevance[query]]
            dr_dests = [d[0].lower() for d in self.dr_results[query_dr][:self.k]]

            rel_scores = []
            dr_scores = [1 for d in self.dr_results[query_dr][:self.k]]


            for d in dr_dests:
                found = False
                for d2, d2_score in rel_dests:
                    if fuzz.ratio(d, d2) > 85:
                        if d != d2:
                            print(f"Matched {d} to {d2} with {fuzz.ratio(d, d2)}% similarity.")
                        rel_scores.append(1)
                        found = True
                        break
                if not found:
                    rel_scores.append(0)
            ap_per_query[query] = average_precision_score(np.array([rel_scores]), np.array([dr_scores]))
            ap_overall += ap_per_query[query]
            
        ap_overall /= total_queries
        return ap_per_query, ap_overall