import argparse, os
from src.Evaluator.PrecisionK import PrecisionK
from src.Evaluator.RPercision import RPrecision
from src.Evaluator.RecallK import RecallK
from src.Evaluator.mAPK import mAPK

EVALUATOR_TYPES = {"precision", "rprecision", "recall", "map"}

def main(args):
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.evaluator == "precision":
        evaluator = PrecisionK(
            ground_truth_path=args.ground_truths,
            ranked_result_path=args.ranked_result_path,
            output_path=output_path,
            k=args.k
        )
    elif args.evaluator == "rprecision":
        evaluator = RPrecision(
            ground_truth_path=args.ground_truths,
            ranked_result_path=args.ranked_result_path,
            output_path=output_path
        )
    elif args.evaluator == "recall":
        evaluator = RecallK(
            ground_truth_path=args.ground_truths,
            ranked_result_path=args.ranked_result_path,
            output_path=output_path,
            k=args.k
        )
    elif args.evaluator == "map":
        evaluator = mAPK(
            ground_truth_path=args.ground_truths,
            ranked_result_path=args.ranked_result_path,
            output_path=output_path,
            k=args.k
        )
    else:
        raise ValueError("Invalid evaluator type. Available types are: {}".format(", ".join(sorted(EVALUATOR_TYPES))))

    evaluator.run_evaluation()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ranked results against ground truth data using specified metrics.")
    parser.add_argument("-e", "--evaluator", type=str, choices=EVALUATOR_TYPES, required=True, 
                        help="Type of evaluator to use. Available options: {}".format(", ".join(sorted(EVALUATOR_TYPES))))
    parser.add_argument("-k", "--k", type=int, default=50, 
                        help="The number of top results to consider for evaluation (applicable for Precision@K, Recall@K, and mAP@K).")
    parser.add_argument("-j", "--ranked_result_path", required=True, 
                        help="Path to the JSON file containing ranked results.")
    parser.add_argument("-g", "--ground_truths", required=True, 
                        help="Path to the JSON file containing ground truth data.")
    parser.add_argument("-o", "--output_path", required=True, 
                        help="Path where the evaluation results will be stored.")
    args = parser.parse_args()
    main(args=args)