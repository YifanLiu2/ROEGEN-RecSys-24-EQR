from src.Evaluator.BaseEvaluator import Evaluator

class RPrecision(Evaluator):
    def __init__(self, ground_truth_path: str, ranked_result_path: str, output_path: str):
        super().__init__(ground_truth_path, ranked_result_path, output_path)

        
    def evaluate(self, ground_truth: list[str], ranked_list: list[str]) -> float:
        """
        """
        if not ground_truth:
            return 0
        
        R = min(len(ground_truth), len(ranked_list))
        relevant_count = sum(1 for doc in ranked_list[:min(R, len(ranked_list))] if doc in ground_truth)
        r_precision = relevant_count / R if R > 0 else 0

        return r_precision