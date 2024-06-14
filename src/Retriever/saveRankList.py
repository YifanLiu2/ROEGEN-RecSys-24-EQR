import argparse
import json

def main(args):
    output_dir = args.output_dir
    results_path = args.results_path
    with open(results_path, "r") as file:
        results = json.load(file)
    count = 0
    # method name is the chunk just before .json
    method_name = results_path.split("_")[-1].split(".")[0]
    output_result = dict()
    for query in results:
        # Sort the destinations by score
        dests = results[query]
        cities_rank_list = []
        sorted_dests = sorted(dests.items(), key=lambda x: x[1][0], reverse=True)
        for dest, (score, aspects) in sorted_dests:
            cities_rank_list.append(dest)
        output_result[query] = cities_rank_list
    output_path = f"{output_dir}/rank_list_{method_name}.json"
    with open(output_path, "w") as file:
        json.dump(output_result, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--results_path", required=True, help="Path to the results file")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to the output file")
    args = parser.parse_args()
    main(args=args)