import json
# preview test results
with open("output/dense_results_total_def_top3.json", "r") as file:
    results = json.load(file)
# with open("output/dense_results_top3_gpt.json", "r") as file:
#     results = json.load(file)

# with open("output/hybrid_results_with_qe_v2_top3.json", "r") as file:
#     results = json.load(file)
# Separate the results by queries
for query in results:
    print(f"Query: {query}")
    # Sort the destinations by score
    results[query] = {k: v for k, v in sorted(results[query].items(), key=lambda item: item[1], reverse=True)}
    # print top 50 destinations
    for dest in list(results[query].keys())[:50]:
        print(f"Destination: {dest}")
        print(f"Score: {results[query][dest][0]}")
        # print(f"Top chunk: {results[query][dest][1]}")
    print("\n")
    # Store top 50 destinations into a json file
    # query name should be cleaned, with only alphanumeric characters
    query_name = "".join([c for c in query if c.isalnum()])
    with open(f"output/{query_name}_top50.json", "w") as file:
        json.dump(list(results[query].keys())[:50], file)

# preview the first query
# query = list(results.keys())[0]
# print(f"Query: {query}")
# # Sort the destinations by score
# results[query] = {k: v for k, v in sorted(results[query].items(), key=lambda item: item[1], reverse=True)}
# # print top 20 destinations
# for dest in list(results[query].keys())[:20]:
#     print(f"Destination: {dest}")
#     print(f"Score: {results[query][dest][0]}")
#     print(f"Top chunk: {results[query][dest][1]}")
# print("\n")
