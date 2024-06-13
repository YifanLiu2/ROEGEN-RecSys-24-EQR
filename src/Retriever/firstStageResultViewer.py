import json
# preview test results
with open("output/dense_results_gpt_elaborate.json", "r") as file:
    results = json.load(file)

# preview the queries
# the format is {query: {dest: (dest_score, {"aspect": top_chunk})}}

for query in results:
    print(f"Query: {query}")
    # Sort the destinations by score
    dests = results[query]
    sorted_dests = sorted(dests.items(), key=lambda x: x[1][0], reverse=True)

    # print the top 50 destinations
    for dest, (score, aspects) in sorted_dests[:50]:
        print(f"Destination: {dest}")
        print(f"Score: {score}")
        # for aspect, top_chunk in aspects.items():
        #     print(f"Aspect: {aspect}")
        #     print(f"Top chunk: {top_chunk}")
        print("\n")
    print("\n")


