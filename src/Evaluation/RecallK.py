from BaseEvaluator import Evaluator
from fuzzywuzzy import fuzz

class RecallK(Evaluator):

    def __init__(self, k: int, relevance, dr_results, query_mapper=None):
        super().__init__(k, relevance, dr_results, query_mapper)

    def recall_at_k(self):
        # get per query and overall and return both
        recall_per_query = {}
        recall_overall = 0
        total_queries = len(self.relevance)
        for query in self.relevance:
            query_dr = query
            if self.query_mapper:
                query_dr = self.query_mapper[query]

            rel_dests = [d[0].lower() for d in self.relevance[query]]
            dr_dests = [d[0].lower() for d in self.dr_results[query_dr][:self.k]]

            correct = 0
            for d in dr_dests:
                for d2 in rel_dests:
                    if fuzz.ratio(d, d2) > 85:
                        if d != d2:
                            print(f"Matched {d} to {d2} with {fuzz.ratio(d, d2)}% similarity.")
                        correct += 1
                        break
            
            recall_per_query[query] = correct / len(rel_dests)
            recall_overall += correct / len(rel_dests)
        
        recall_overall /= total_queries
        return recall_per_query, recall_overall