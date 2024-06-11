import pickle

with open ("output/processed_query.pkl", "rb") as file:
    queries = pickle.load(file)

# preview the first query
query = queries[4]
print(f"Query: {query.description}")
# print descriptions
for description in query.get_descriptions():
    print(f"Description: {description}")
# Get all constraints
for constraint in query.constraints:
    print(f"Constraint: {constraint.definition}")