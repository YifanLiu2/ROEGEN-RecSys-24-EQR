import argparse
import os
import json
from src.Evaluator.PrecisionK import PrecisionK
from src.Evaluator.RPercision import RPrecision
from src.Evaluator.RecallK import RecallK
from src.Evaluator.mAPK import mAPK
TYPE={"precision", "rprecision", "recall", "map"}
def main(args):

    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.evaluator == "precision":
        evaluator = PrecisionK(ground_truth_path=args.ground_truths, ranked_result_path=args.ranked_result_path, output_path=output_path, k=args.k)
    elif args.evaluator == "rprecision":
        evaluator = RPrecision(ground_truth_path=args.ground_truths, ranked_result_path=args.ranked_result_path, output_path=output_path)
    elif args.evaluator == "recall":
        evaluator = RecallK(ground_truth_path=args.ground_truths, ranked_result_path=args.ranked_result_path, output_path=output_path, k=args.k)
    elif args.evaluator == "map":
        evaluator = mAPK(ground_truth_path=args.ground_truths, ranked_result_path=args.ranked_result_path, output_path=output_path, k=args.k)
    else:
        raise ValueError("Invalid evaluator type")
    
    evaluator.run_evaluation()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluator.")
    parser.add_argument("-e", "--evaluator", type=str, choices=TYPE, required=True, help="Specify the type of the evaluator. Available types are: {}".format(", ".join(sorted(TYPE))))
    parser.add_argument("-k", "--k", type=int, help="Top k results to consider", default=50)
    parser.add_argument("-j", "--ranked_result_path", required=True, help="Path to the json file containing the results")
    parser.add_argument("-g", "--ground_truths", required=True, help="Directory to the json file containing the ground truths")
    parser.add_argument("-o", "--output_path", required=True, help="Path to store the output")
    args = parser.parse_args()
    main(args=args)