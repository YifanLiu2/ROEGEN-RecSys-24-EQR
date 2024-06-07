class Evaluator:
    def __init__(self, json_path, ground_truths: dict, k : int = None):
        self.k = k
        self.json_path = json_path
        self.ground_truths = ground_truths