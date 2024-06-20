from src.Evaluator.BaseEvaluator import Evaluator

class RecallK(Evaluator):
    def __init__(self, ground_truth_path: str, ranked_result_path: str, output_path: str, k: int):
        super().__init__(ground_truth_path, ranked_result_path, output_path)
        self.k = k

    def evaluate(self, ground_truth: list[str], ranked_list: list[str]) -> float:
        """
        """
        k = min(self.k, len(ranked_list))
        top_k_items = set(ranked_list[:k])
        relevant_in_top_k = len(top_k_items.intersection(set(ground_truth)))
        total_relevant = len(set(ground_truth))
        
        if total_relevant == 0:
            return 0 
        else:
            return relevant_in_top_k / total_relevant