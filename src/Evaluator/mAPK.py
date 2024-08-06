from src.Evaluator.BaseEvaluator import Evaluator

class mAPK(Evaluator):
    def __init__(self, ground_truth_path: str, ranked_result_path: str, output_path: str, k: int):
        super().__init__(ground_truth_path, ranked_result_path, output_path)
        self.k = k

    def evaluate(self, ground_truth: list[str], ranked_list: list[str]) -> float:
        if not ground_truth:
            return 0  

        hit_count = 0
        sum_precisions = 0
        relevant_found = 0

        k = min(self.k, len(ranked_list))

        for index, item in enumerate(ranked_list[:k]):
            if item in ground_truth:
                relevant_found += 1
                hit_count += 1
                precision_at_index = hit_count / (index + 1)
                sum_precisions += precision_at_index

        if relevant_found > 0:
            return sum_precisions / relevant_found
        else:
            return 0 