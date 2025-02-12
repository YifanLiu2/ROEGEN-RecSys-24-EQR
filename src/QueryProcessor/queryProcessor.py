import json, os, pickle
from tqdm import tqdm
from src.LLM.GPTChatCompletion import *
from src.Entity.query import *

ANSWER_FORMAT = """{{"answer": {answer}}}"""

RETRIEVER_TYPE = {"dense", "sparse"}

LIST_RESPONSE = {
    "type": "json_schema",
    "json_schema": {
    "name": "answers",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
        "answer": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
        },
        "required": ["answer"],
        "additionalProperties": False
    }
    }
}

STRING_RESPONSE = {
    "type": "json_schema",
    "json_schema": {
    "name": "answers",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
        "answer": {
            "type": "string",
        }
        },
        "required": ["answer"],
        "additionalProperties": False
    }
    }
}



class QueryProcessor:
    """
    A class to reformulate queries using a specified LLM for a retrieval method. 
    """
    cls_type = "none"
    def __init__(self, input_path: str, llm: LLM, retriever_type: str = "dense", output_dir: str = "output"):
        self.llm = llm
        self.output_dir = output_dir
        self.query_list = self._load_queries(input_path)
        self.retriever_type = retriever_type
    

    def process_query(self) -> list[AbstractQuery]:
        """
        Reformulate the queries.
        """
        result_queries = []
        for query in tqdm(self.query_list, desc="Processing queries", unit="query"):

            curr_query = AbstractQuery(description=query)
            
            new_desc = self.reformulate_query(query=curr_query)
            curr_query.set_reformulation(reformulation=new_desc)
            
            result_queries.append(curr_query)

        self._save_results(result=result_queries)
        return result_queries
    

    def _load_queries(self, input_path: str) -> list[str]:
        """
        Loads queries from a text file.

        input_path (str): The path to the input file containing queries.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f]
        return queries
    

    def _save_results(self, result: list[AbstractQuery]):
        """
        Saves the processed query results

        result (list[AbstractQuery]): A list of AbstractQuery objects to be saved.
        """
        pkl_path = os.path.join(self.output_dir, f"processed_query_{self.cls_type}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        query_data = []
        for query in result:
            query_info = {
                "Query": query.get_description(),
                "Reformulation": query.get_reformulation(),
            }
            query_data.append(query_info)

        json_path = os.path.join(self.output_dir, f"processed_query_{self.cls_type}.json")
        with open(json_path, 'w') as f:
            json.dump(query_data, f, indent=4)
        

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        Reformulates the description of a given query.

        query (AbstractQuery): The query to be reformulated.
        """
        return query.get_description()
    

class GQR(QueryProcessor):
    """
    GQR
    """
    cls_type = "gqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense"):
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)

    def reformulate_query(self, query: AbstractQuery) -> str:
        prompt = """
        Paraphrase the following query: {query} Paraphrase:
        Provide your answers in JSON format: {{"answer": "YOUR ANSWER"}}.
        """
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a recommendation expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations")},
            {"role": "assistant", "content": answer.format(answer="Best Cities for a Family Vacation")},
            {"role": "user", "content": prompt.format(query="Which hotels are popular with repeat visitors?")},
            {"role": "assistant", "content": answer.format(answer="Which hotels are frequently chosen by returning guests?")},
            {"role": "user", "content": prompt.format(query="Eateries featuring seasonal menus, locally sourced ingredients, and farm-to-table philosophies.")},
            {"role": "assistant", "content": answer.format(answer="Restaurants that highlight seasonal dishes, use locally sourced ingredients, and embrace farm-to-table dining.")},          
            {"role": "user", "content": prompt.format(query=query.get_description())},
        ]
        response = self.llm.generate(message, response_format=STRING_RESPONSE)
        paraphrase = json.loads(response)["answer"]
        
        query_str = query.get_description()
        paraphrase_list = [query_str] + [paraphrase]

        if self.retriever_type == "sparse":    
            paraphrases = ' '.join(paraphrase_list)
        
        else: # for dense retriever
            paraphrases = '[SEP]'.join(paraphrase_list)

        return paraphrases


class Q2E(QueryProcessor):
    """
    Q2E
    """
    cls_type = "q2e"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense"):
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)

    def reformulate_query(self, query: AbstractQuery) -> str:
        prompt = """
        Write a list of keywords for the given query: Query: {query} Keywords:
        Provide your answers in JSON format: {{"answer": [LIST]}}.
        """
        answer = ANSWER_FORMAT

        # Few-shot examples
        message = [
            {"role": "system", "content": "You are a recommendation expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations")},
            {"role": "assistant", "content": answer.format(answer=["theme parks", "child-friendly museums", "beach resorts", "educational tours", "outdoor adventures"])},
            {"role": "user", "content": prompt.format(query="Which hotels are popular with repeat visitors?")},
            {"role": "assistant", "content": answer.format(answer=["guest loyalty programs", "high customer ratings", "frequent traveler perks", "boutique hotels", "chain hotels", "business traveler amenities", "luxury resorts", "all-inclusive stays", "personalized service", "long-term stay discounts"])},          
            {"role": "user", "content": prompt.format(query="Eateries featuring seasonal menus, locally sourced ingredients, and farm-to-table philosophies.")},
            {"role": "assistant", "content": answer.format(answer=["farm-to-table restaurants", "seasonal menus", "locally sourced ingredients", "organic dining", "sustainable cuisine", "chef-driven menus", "artisanal food", "slow food movement", "regional specialties", "farmers market restaurants"])},          
            {"role": "user", "content": prompt.format(query=query.get_description())},
        ]

        response = self.llm.generate(message, response_format=LIST_RESPONSE)
        expansion_list = json.loads(response)["answer"]

        query_str = query.get_description()
        expansion_list = [query_str] + expansion_list

        if self.retriever_type == "sparse":
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansions = '[SEP]'.join(expansion_list)

        return expansions


class Q2D(QueryProcessor):
    """
    Q2D
    """
    cls_type = "q2d"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense"):
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)

    def reformulate_query(self, query: AbstractQuery) -> str:
        prompt = """
        Write a passage that answers the given query: Query: {query} Passage:
        Provide your answers in JSON format: {{"answer": "YOUR ANSWER"}}.
        """
        answer = ANSWER_FORMAT

        # Few-shot examples
        message = [
            {"role": "system", "content": "You are a recommendation expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations")},
            {"role": "assistant", "content": answer.format(answer="When planning a family-friendly vacation, choosing the right city can make all the difference in ensuring a memorable and enjoyable experience for everyone. Orlando, Florida, USA, known as the theme park capital of the world, offers attractions like Walt Disney World Resort, Universal Studios, and the Kennedy Space Center. San Diego, California, USA, is home to beautiful beaches, the world-famous San Diego Zoo, and Legoland California. Tokyo, Japan, offers Tokyo Disneyland and DisneySea, along with historic temples and delicious Japanese cuisine. These cities provide a range of attractions and activities designed to entertain and educate family members of all ages, ensuring a vacation filled with fun, learning, and unforgettable memories.")},
            {"role": "user", "content": prompt.format(query="Which hotels are popular with repeat visitors?")},
            {"role": "assistant", "content": answer.format(answer="When looking for hotels that are popular with repeat visitors, certain accommodations stand out due to their exceptional service, comfort, and amenities that keep guests coming back. The Ritz-Carlton in Tokyo, Japan, is renowned for its luxurious rooms, breathtaking city views, and outstanding hospitality. The Waldorf Astoria in New York City, USA, is a historic hotel known for its elegant design and world-class service. In Paris, France, Le Meurice attracts repeat guests with its blend of classic charm and modern luxury, located near the Louvre Museum. Additionally, family-friendly resorts like the Grand Floridian Resort & Spa at Walt Disney World in Florida offer immersive experiences and top-tier service that encourage return visits. These hotels consistently receive high ratings from travelers who appreciate their attention to detail, premium amenities, and welcoming atmosphere.")},          
            {"role": "user", "content": prompt.format(query="Eateries featuring seasonal menus, locally sourced ingredients, and farm-to-table philosophies.")},
            {"role": "assistant", "content": answer.format(answer="Eateries that focus on seasonal menus, locally sourced ingredients, and farm-to-table philosophies offer a dining experience that highlights freshness, sustainability, and regional flavors. In California’s Napa Valley, restaurants like The French Laundry showcase farm-fresh produce and artisanal ingredients, creating exquisite seasonal dishes. In Portland, Oregon, Farm Spirit offers a plant-based menu sourced entirely from the Pacific Northwest, ensuring a deep connection to local agriculture. Copenhagen, Denmark, home to Noma, revolutionized the concept of seasonal dining by emphasizing foraged ingredients and Nordic culinary traditions. These eateries provide an ever-changing menu that reflects the best of each season, offering diners a unique and flavorful experience rooted in local terroir and sustainable practices.")},          
            {"role": "user", "content": prompt.format(query=query.get_description())},
        ]

        # parse the answer
        response = self.llm.generate(message, response_format=STRING_RESPONSE)
        pseudo_doc = json.loads(response)["answer"]

        query_str = query.get_description()
    

        if self.retriever_type == "sparse":
            expansion_list = [query_str] + [pseudo_doc]
            expansions = ' '.join(expansion_list)
        
        else:
            expansion_list = [query_str] + [pseudo_doc]
            expansions = '[SEP]'.join(expansion_list)
        
        return expansions


class EQR(QueryProcessor):
    """
    EQR
    """
    cls_type = "eqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense", k: int = 5):
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)
        self.k = k

    def reformulate_query(self, query: AbstractQuery) -> str:
        answer = ANSWER_FORMAT

        prompt = """
        Query: {query}

        This query requests recommendations from our multi-domain system (e.g., hotels, restaurants, cities, etc.). First, clearly specify which type(s) of recommendations you are addressing based on the query without any explanation in your response.

        Then, please do the following:
        1. Break down the query into {k} distinct subtopics or aspects.
        2. For each subtopic, provide:
            - A one-sentence explanation that clarifies the subtopic.
            - Example items or recommendations relevant to that subtopic.
        3. Provide your answers in JSON format: {{"answer": [LIST]}}.
        """
        
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a recommendation expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=5)},
            {"role": "assistant", "content": answer.format(answer=["Theme Parks - Cities with expansive theme parks offering thrilling rides and attractions suitable for all ages such as Orlando and Los Angeles.", "Zoos and Aquariums - Feature diverse collections of animals and underwater displays such as Chicago and San Diego.", "Children Museums - Tailored for younger visitors with hands-on learning exhibits such as Indianapolis and New York City.", "Beaches - Safe, clean beaches with gentle waves and family amenities such as Honolulu and Miami.", "Parks - Large parks with playgrounds, picnic areas, and public events such as London and Vancouver."])},
            {"role": "user", "content": prompt.format(query="Which hotels are popular with repeat visitors?", k=5)},
            {"role": "assistant", "content": answer.format(answer=[
                "Customer Loyalty and Repeat Stays - Hotels with a high percentage of returning guests due to exceptional service and strong loyalty programs, such as Marriott Bonvoy Hotels and Hilton Honors Hotels.",
                "Service Quality and Guest Satisfaction - Hotels known for high ratings in cleanliness, customer service, and overall experience, such as Ritz-Carlton and Four Seasons.",
                "Value for Money - Hotels that provide excellent amenities and comfort at reasonable prices, attracting repeat visitors, such as Hampton Inn and Holiday Inn Express.",
                "Location Convenience - Hotels situated near business districts, attractions, or transport hubs, making them appealing for repeat stays, such as Park Hyatt Tokyo and The Peninsula Chicago.",
                "Exclusive Membership and Rewards - Hotels with robust loyalty programs offering perks like free stays, upgrades, and discounts, such as IHG One Rewards and Accor Live Limitless."
            ])},
            {"role": "user", "content": prompt.format(query="Eateries featuring seasonal menus, locally sourced ingredients, and farm-to-table philosophies.", k=5)},
            {"role": "assistant", "content": answer.format(answer=[
                "Seasonal Menus - Restaurants that adapt their menus based on seasonal availability, offering fresh and unique dishes throughout the year, such as Chez Panisse in Berkeley and Blue Hill at Stone Barns in New York.",
                "Locally Sourced Ingredients - Eateries that prioritize ingredients from nearby farms and producers, ensuring freshness and supporting local agriculture, such as Husk in Charleston and Noma in Copenhagen.",
                "Farm-to-Table Dining - Restaurants that maintain a direct relationship with farms, focusing on transparency and sustainability in their food sourcing, such as The French Laundry in Napa Valley and Farmstead at Long Meadow Ranch in St. Helena.",
                "Sustainable and Ethical Practices - Establishments committed to ethical sourcing, organic produce, and environmentally friendly dining, such as Eleven Madison Park in New York and SingleThread in Healdsburg.",
                "Community-Driven Culinary Experiences - Restaurants deeply integrated with local food movements and regional culinary traditions, emphasizing community engagement, such as Dandelion Market in Charlotte and The Willows Inn on Lummi Island."
            ])},
            {"role": "user", "content": prompt.format(query=query.get_description(), k=self.k)},
        ]

        response = self.llm.generate(message, response_format=LIST_RESPONSE)
        expansion_list = json.loads(response)["answer"]
        
        query_str = query.get_description()
        expansion_list = [query_str] + expansion_list
        
        if self.retriever_type == "sparse":            
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansions = '[SEP]'.join(expansion_list)

        return expansions