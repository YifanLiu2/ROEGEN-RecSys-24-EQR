import abc, json
from tqdm import tqdm

class Evaluator(abc.ABC):
    def __init__(self, ground_truth_path: str, ranked_result_path: str, output_path: str):
        with open(ground_truth_path, encoding="utf-8") as f:
            ground_truth = json.load(f)
        
        with open(ranked_result_path, encoding="utf-8") as f:
            ranked_result = json.load(f)
        
        self.ground_truth = ground_truth
        self.ranked_result = ranked_result
        self.output_path = output_path
    
    def run_evaluation(self):
        """
        """
        results = {}
        for query, ranked_list in tqdm(self.ranked_result.items(), desc="Evaluating queries"):
            # check if query exist
            if query not in self.ground_truth:
                print(f"Ground truth not exist for the query: {query}")

            ground_truth = self.ground_truth[query]
            
            # run evaluation
            score = self.evaluate(ground_truth, ranked_list)
            results[query] = score

        with open(self.output_path, 'w', encoding="utf-8") as f:
            json.dump(results, f, indent=4)


    @abc.abstractmethod
    def evaluate(self, ground_truth: list[str], ranked_list: list[str]) -> float:
        """
        Abstract method to be implemented for evaluation.
        """
        pass