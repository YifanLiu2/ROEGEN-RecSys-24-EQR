import json
import argparse
# preview test results
def main(args):
    results_path = args.results_path
    top_cities = args.top_cities
    top_chunks = args.top_chunks
    with open(results_path, "r") as file:
        results = json.load(file)

    # preview the queries
    # the format is {query: {dest: (dest_score, {"aspect": top_chunk})}}

    for query in results:
        print(f"Query: {query}")
        # Sort the destinations by score
        dests = results[query]
        sorted_dests = sorted(dests.items(), key=lambda x: x[1][0], reverse=True)

        # print the top 50 destinations
        for dest, (score, aspects) in sorted_dests[:top_cities]:
            print(f"Destination: {dest}")
            print(f"Score: {score}")
            if top_chunks:
                for aspect, top_chunk in aspects.items():
                    print(f"Aspect: {aspect}")
                    print(f"Top chunk: {top_chunk}")
            print("\n")
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the dense retriever.")
    parser.add_argument("-r", "--results_path", required=True, help="Path to the results file")
    parser.add_argument("-t", "--top_cities", help="Number of top cities to display", default=50)
    parser.add_argument("-c", "--top_chunks", help="whether to display top chunks", default=False)
    args = parser.parse_args()
    main(args=args)
