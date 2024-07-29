import os, json, argparse
from tqdm import tqdm
from src.Entity.query import AbstractQuery
from src.Retriever.abstractRetriever import *
from src.LLM.GPTChatCompletion import *
from config import API_KEY

with open("data/total_cities.txt", encoding="utf-8") as f:
    CITY_LIST = f.readlines()


class LLMRetriever(AbstractRetriever):
    """
    """
    def __init__(self, query_path: str, output_dir: str, llm: LLM = GPTChatCompletion(api_key=API_KEY), k: int = 100):
        self.query_path = query_path
        self.output_dir = output_dir
        self.llm = llm
        self.k = k
    
    def retrieval_for_dest(self):
        pass

    def retrieval_for_query(self, query: str) -> list[str]:
        prompt = """
        Query: {query}
        Task: Generate a ranked list of {k} cities from a provided city list based on a user's travel recommendation query. Ensure that:
            - Rate each city on a scale from 1 to 10 according to their relevance to the query. Rank the cities based on these ratings in descending order and select only the top {k}.
            - The final ranked list should be descendingly sorted, including only city names, with the most relevant city at the top.
            - Manually check whether your example cities come from the choice of cities, if not, replace with a new one in the list.
            - Manually check whether the provided list has length {k}, if not, add or remove cities to the list.
            - Present your results in valid JSON format with double quotes: {{"answer": [LIST]}}, where [LIST] is an array of valid city names from the provided city list.
        
        This is your choice of cities:
        {cities}
        """  
        cities = ",".join(CITY_LIST)

        message = [
            {"role": "system", "content": "You are a travel expert."},
            
            {"role": "user", "content": prompt.format(query=query, cities=cities, k=self.k)},
        ]

        try:
            response = self.llm.generate(message)
            # parse the answer
            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
            ranked_list = json.loads(response)["answer"]
        
        except Exception as e:
            print(f"Failed to generate ranked list for query '{query}': {e}")
            return []

        if not isinstance(ranked_list, list):
            print(f"Invalid ranked list for query '{query}': {ranked_list}")
            return []

        return ranked_list

    
    def run_retrieval(self):
        """
        Loads necessary data and runs the dense retrieval process, then saves the results to the specified output path.
        """
        queries = self.load_queries()

        results = dict()
        for query in tqdm(queries, desc="Processing queries"):
            query_str = query.get_description() if isinstance(query, AbstractQuery) else query
            
            ranked_list = self.retrieval_for_query(query=query)
            results[query_str] = ranked_list

        ranked_list_path = os.path.join(self.output_dir, "ranked_list.json")
        
        # save dense result
        with open(ranked_list_path, "w") as file_ranked:
            json.dump(ranked_list, file_ranked, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LLM-based recommender")
    parser.add_argument('--query_path', type=str, required=True, help='Path to the JSON file containing queries')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the retrieval results will be saved')
    parser.add_argument('--k', type=int, default=100, help='')
    args = parser.parse_args()

    retriever = LLMRetriever(query_path=args.query_path, output_dir=args.output_dir, k=args.k)
    retriever.run_retrieval()
    
