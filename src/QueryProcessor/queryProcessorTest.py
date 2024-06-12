import argparse
from config import API_KEY
from src.QueryProcessor.queryProcessor import *
from src.LLM.GPTChatCompletion import *

def main(parser):
    llm = GPTChatCompletion(api_key=API_KEY)
    input_path = parser.input_path
    mode_name = parser.mode_name
    output_dir = parser.output_dir
    query_processor = queryProcessor(input_path=input_path, llm=llm, mode_name=mode_name, output_dir=output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some queries using an LLM.")
    parser.add_argument("input_path", help="Path to the input file containing queries")
    parser.add_argument("mode_name", choices=MODE, help="Processing mode (choose from: {})".format(", ".join(MODE)))
    parser.add_argument("output_dir", help="Directory to store processed queries")
    main(parser=parser)


# init GPT-4 model
llm = GPTChatCompletion(api_key=API_KEY)

# init query processor
# modle name can only be expand, reformulate or elaborate
mode_name = "elaborate"


# process queries
query_lists = query_processor.process_query() # returns a list of query classes


# preview the entire processed queries
for query in query_lists:
    print(f"Query: {query.get_description()}")
    for preference in query.preferences:
        print(f"Original text: {preference.get_original_description()}")
        print(f"Preference: {preference.get_new_description()}")
    for constraint in query.constraints:
        print(f"Original text: {constraint.get_original_description()}")
        print(f"Constraint: {constraint.get_new_description()}")
    for hybrid in query.hybrids:
        print(f"Original text: {hybrid.get_original_description()}")
        print(f"Hybrid: {hybrid.get_new_description()}")
    print("\n")
