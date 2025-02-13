import argparse, os
from config import API_KEY
from src.QueryProcessor.queryProcessor import *
from src.LLM.GPTChatCompletion import *

MODE = {"eqr", "gqr", "q2d", "q2e", "none"}
RETRIEVER_TYPE = {"sparse", "dense"}

def main(args):
    llm = GPTChatCompletion(api_key=API_KEY)
    input_path = args.input_path
    mode_name = args.mode
    output_dir = args.output_dir
    retriever_type = args.retriever_type
    k = args.k

    if k and k <= 0:
        raise ValueError("Invalid k, must be a positive integer: {k}")

    if not input_path.endswith('.txt'):
        raise ValueError(f"Invalid file type: {input_path} is not a .txt file")
    
    os.makedirs(output_dir, exist_ok=True)

    if not mode_name or mode_name == "none":
        query_processor = QueryProcessor(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type, mode="none")
    
    elif mode_name == "gqr":
        query_processor = GQR(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type, mode=mode_name)
    
    elif mode_name == "q2e":
        query_processor = Q2E(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type, mode=mode_name)
        
    elif mode_name == "q2d":
        query_processor = Q2D(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type, mode=mode_name)
    
    elif "eqr" in mode_name:
        if isinstance(mode_name.split("_")[-1], int):
            k = mode_name.split("_")[-1]

        elif not k:
            raise ValueError("Must specify k for {mode_name} method")
        query_processor = EQR(input_path=input_path, llm=llm, output_dir=output_dir, k=k, retriever_type=retriever_type, mode=mode_name)

    query_processor.process_query()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a query reformulation pipeline using a LLM.")
    parser.add_argument("--input_path", required=True, help="Path to the input .txt file containing queries.")    
    parser.add_argument("--output_dir", help="Directory to store processed queries", default="output")
    parser.add_argument("--mode", required=False, help="Query reformulation method. Available options: {}.".format(", ".join(sorted(MODE))))
    parser.add_argument("--retriever_type", choices=sorted(RETRIEVER_TYPE), default="dense", help="Type of retriever to use. Available options: {}.".format(", ".join(sorted(RETRIEVER_TYPE))))
    parser.add_argument("--k", default=5, type=int, help="Number of top results to process, required for 'eqr'")

    args = parser.parse_args()
    main(args=args)


