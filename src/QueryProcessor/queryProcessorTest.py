import pickle
from config import API_KEY
from src.QueryProcessor.queryProcessor import *
from src.LLM.GPTChatCompletion import *

# init query
# query_string_list = ["Can you recommend cities with Disney attractions for my next vacation?",
#                      "which cities are known for being safe and welcoming for people traveling alone?",
#                      "I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there?",
#                      "What cities in Europe host cultural festivals during the summer months that I shouldn't miss?",
#                      "Seeking cities in tropical region suitable for a family vacation with kids."]
query_string_list = ["As a solo traveler, which cities are known for being safe and welcoming for people traveling alone?"]

# init GPT-4 model
llm = GPTChatCompletion(api_key=API_KEY)

# init query processor
# modle name can only be expand, reformulate or elaborate
mode_name = "elaborate"
query_processor = queryProcessor(query=query_string_list, llm=llm, mode_name=mode_name, output_dir="output")

# process queries
query_lists = query_processor.process_query() # returns a list of query classes

# save processed queries into a pickle file
with open(f"output/processed_query_{mode_name}.pkl", "wb") as file:
    pickle.dump(query_lists, file)

# preview the entire processed queries
for query in query_lists:
    print(f"Query: {query.get_description()}")
    for preference in query.preferences:
        print(f"Preference: {preference.get_new_description()}")
    for constraint in query.constraints:
        print(f"Constraint: {constraint.get_new_description()}")
    print("\n")
