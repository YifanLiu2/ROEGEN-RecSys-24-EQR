import argparse, os
from config import API_KEY
from src.QueryProcessor.queryProcessor import *
from src.LLM.GPTChatCompletion import *

MODE = {"expand", "reformulate", "elaborate"}

def main(args):
    llm = GPTChatCompletion(api_key=API_KEY)
    input_path = args.input_path
    mode_name = args.mode_name
    output_dir = args.output_dir

    if not input_path.endswith('.txt'):
        raise ValueError(f"Invalid file type: {input_path} is not a .txt file")
    
    os.makedirs(output_dir, exist_ok=True)

    query_processor = queryProcessor(input_path=input_path, llm=llm, mode_name=mode_name, output_dir=output_dir)
    query_list = query_processor.process_query()

    # for query in query_list:
    #     print(f"Query: {query.get_description()}")
    #     for preference in query.preferences:
    #         print(f"Original text: {preference.get_original_description()}")
    #         print(f"Preference: {preference.get_new_description()}")
    #     for constraint in query.constraints:
    #         print(f"Original text: {constraint.get_original_description()}")
    #         print(f"Constraint: {constraint.get_new_description()}")
    #     for hybrid in query.hybrids:
    #         print(f"Original text: {hybrid.get_original_description()}")
    #         print(f"Hybrid: {hybrid.get_new_description()}")
    #     print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some queries using an LLM.")
    parser.add_argument("-i", "--input_path", required=True, help="Path to the input file containing queries")
    parser.add_argument("-m", "--mode_name", required=True, help="Processing mode")
    parser.add_argument("-o", "--output_dir", help="Directory to store processed queries", default="output")
    args = parser.parse_args()
    main(args=args)


