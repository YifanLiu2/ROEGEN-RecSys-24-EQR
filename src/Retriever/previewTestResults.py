import json
# preview test results
# output/dense_results_with_qe.json
with open("output/hybrid_results.json", "r") as file:
    results = json.load(file)
# Separate the results by queries
for query in results:
    print(f"Query: {query}")
    # Sort the destinations by score
    results[query] = {k: v for k, v in sorted(results[query].items(), key=lambda item: item[1], reverse=True)}
    # print top 20 destinations
    for dest in list(results[query].keys())[:5]:
        print(f"Destination: {dest}")
        print(f"Score: {results[query][dest][0]}")
        print(f"Top chunk: {results[query][dest][1]}")
    print("\n")