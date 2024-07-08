import argparse, os
from config import API_KEY
from src.QueryProcessor.queryProcessor import *
from src.LLM.GPTChatCompletion import *

MODE = {"gqr","q2d", "q2e", "genqr", "elaborate", "answer", "none"}

def main(args):
    llm = GPTChatCompletion(api_key=API_KEY)
    input_path = args.input_path
    mode_name = args.mode
    if mode_name == "none":
        mode_name = None
    output_dir = args.output_dir

    if not input_path.endswith('.txt'):
        raise ValueError(f"Invalid file type: {input_path} is not a .txt file")
    
    os.makedirs(output_dir, exist_ok=True)

    query_processor = queryProcessor(input_path=input_path, llm=llm, mode_name=mode_name, output_dir=output_dir)
    query_processor.process_query()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some queries using an LLM.")
    parser.add_argument("-i", "--input_path", required=True, help="Path to the input file containing queries")    
    parser.add_argument("-o", "--output_dir", help="Directory to store processed queries", default="output")
    parser.add_argument("--mode", required=False, default=None, choices=MODE, help="Processing mode (choose from: {})".format(", ".join(sorted(MODE))))
    args = parser.parse_args()
    main(args=args)


