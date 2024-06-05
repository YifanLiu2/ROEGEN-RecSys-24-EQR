class Evaluator:
    def __init__(self, k: int, relevance, dr_results, query_mapper = None):
        self.k = k
        self.relevance = relevance
        self.dr_results = dr_results
        self.query_mapper = query_mapper

    