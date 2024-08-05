import json, os, re, pickle
from tqdm import tqdm

from src.LLM.GPTChatCompletion import *
from src.Entity.query import *


ANSWER_FORMAT = """{{"answer": {answer}}}"""
RETRIEVER_TYPE = {"dense", "sparse"}

with open("data/total_cities.txt", encoding="utf-8") as f:
    CITY_LIST = f.readlines()

class QueryProcessor:
    """
    Query Processor class
    """
    cls_type = "none"
    def __init__(self, input_path: str, llm: LLM, retriever_type: str = "dense", output_dir: str = "output"):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        self.llm = llm
        self.output_dir = output_dir
        self.query_list = self._load_queries(input_path)
        self.retriever_type = retriever_type
    

    def process_query(self) -> list[AbstractQuery]:
        """
        Process the queries
        """
        result_queries = []
        for query in tqdm(self.query_list, desc="Processing queries", unit="query"):

            curr_query = AbstractQuery(description=query)
            
            new_desc = self.reformulate_query(query=curr_query)
            curr_query.set_reformuation(reformulation=new_desc)
            
            result_queries.append(curr_query)

        self._save_results(result=result_queries)
        return result_queries
    

    def _load_queries(self, input_path: str) -> list[str]:
        """
        """
        with open(input_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f]
        return queries
    

    def _save_results(self, result: list[AbstractQuery]):
        """
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
        return query.get_description()
    

class GQR(QueryProcessor):
    cls_type = "gqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense", k: int = 1):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)
        self.k = k

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        GQR @ https://dl.acm.org/doi/pdf/10.1145/3589335.3651945
        """
        prompt = """
        Given a user's travel cities recommendation query, generate a list of {k} one longer sentence paraphrase of the query aiming to capture the actual query intent behind the query. 
        Provide your answers in valid JSON format with double quote: {{"answer": [LIST]}}, where paraphrase list is a list of string.
        
        query: {query}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=1)},
            {"role": "assistant", "content": answer.format(answer= ["Cities equipped with an array of child-friendly attractions such as theme parks, interactive museums, and safe, welcoming parks, perfect for family getaways."])},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", k=1)},
            {"role": "assistant", "content": answer.format(answer= ["Cities that provide diverse photographic opportunities, from dramatic urban skylines and historical architecture to vibrant street scenes and serene natural landscapes."])},
            {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", k=1)},
            {"role": "assistant", "content": answer.format(answer=["Destinations celebrated for their extensive horseback riding trails across beaches, mountains, and countrysides, complemented by vibrant cultural events like rodeos, offering a holistic equestrian experience."])},          
            {"role": "user", "content": prompt.format(query=query.get_description(), k=self.k)},
        ]
        response = self.llm.generate(message)

        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        paraphrase_list = json.loads(response)["answer"]
        
        query_str = query.get_description()

        if self.retriever_type == "sparse":
            # append original query description
            query_list = [query_str]
            paraphrase_list = query_list + paraphrase_list
            paraphrases = ' '.join(paraphrase_list)
        
        else: # for dense retriever
            paraphrase_list = [query_str] + paraphrase_list
            paraphrases = '[SEP]'.join(paraphrase_list)

        return paraphrases


class Q2E(QueryProcessor):
    cls_type = "q2e"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense", k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)
        self.k = k

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        Q2E @ https://arxiv.org/pdf/2305.03653
        """
        prompt = """
        Given a user's travel cities recommendation query, break it down the query into {k} distinct keywords related to the given user query aspect.
        Provide your answers in valid JSON format with double quote: {{"answer": [LIST]}}, where keyword list is a list of string.

        query: {query}
        """
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=5)},
            {"role": "assistant", "content": answer.format(answer=["theme parks", "child-friendly museums", "beach resorts", "educational tours", "outdoor adventures"])},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", k=5)},
            {"role": "assistant", "content": answer.format(answer=["coastal views", "natural landscape photography", "exotic island views", "architectural photography", "skyline photography"])},          
            {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", k=5)},
            {"role": "assistant", "content": answer.format(answer=["equestrian", "trail riding", "dressage", "rodeo events", "pony treks"])},          
            {"role": "user", "content": prompt.format(query=query.get_description(), k=self.k)},
        ]

        response = self.llm.generate(message)
        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        expansion_list = json.loads(response)["answer"]

        query_str = query.get_description()

        if self.retriever_type == "sparse":
            # append original query description
            query_list = [query_str]
            expansion_list = query_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)

        return expansions
    

class GenQREnsemble(QueryProcessor):
    cls_type = "genqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense", n: int = 5, k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)
        self.n = n
        self.k = k

    def reformulate_query(self, query: AbstractQuery):
        """
        GenQREnsemble @ https://arxiv.org/pdf/2404.03746
        """
        def rewrite_instruction(instruction: str, n: int):
            """
            """
            prompt = """
            Given the specified user instruction, paraphrase it to a list of {n} instructions. 
            Provide your answers in valid JSON format with double quote: {{"answer": []}}.

            query: {instruction}
            """
            message = [
                {"role": "system", "content": "You are a travel expert."},
                {"role": "user", "content": prompt.format(instruction=instruction, n=n)},
            ]

            response = self.llm.generate(message)

            # parse the answer
            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
            paraphrase_list = json.loads(response)["answer"]
            return paraphrase_list

        instruction = f"Given a user's travel cities recommendation query, break it down the query into {self.k} distinct keywords related to the given user query aspect."
        instruction_list = rewrite_instruction(instruction=instruction, n=self.n)

        answer = ANSWER_FORMAT
        expansion_list = []
        for instruct in instruction_list:
            prompt = """
            {instruct}
            Provide your answers in valid JSON format with double quote: {{"answer": []}}. 
            query: {query}
            """
            message = [
                {"role": "system", "content": "You are a travel expert."},
                {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", instruct="Given a user's travel cities recommendation query, break it down the query into {k} distinct keywords related to the given user query aspect.")},
                {"role": "assistant", "content": answer.format(answer=["theme parks", "child-friendly museums", "beach resorts", "educational tours", "outdoor adventures"])},
                {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", instruct="Given a user's travel cities recommendation query, break it down the query into {k} distinct keywords related to the given user query aspect.")},
                {"role": "assistant", "content": answer.format(answer=["coastal views", "natural landscape photography", "exotic island views", "architectural photography", "skyline photography"])},          
                {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", instruct="Given a user's travel cities recommendation query, break it down the query into {k} distinct keywords related to the given user query aspect.")},
                {"role": "assistant", "content": answer.format(answer=["equestrian", "trail riding", "dressage", "rodeo events", "pony treks"])}, 
                {"role": "user", "content": prompt.format(query=query.get_description(), instruct=instruct)},
            ]
            
            response = self.llm.generate(message)

            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
            results = json.loads(response)["answer"]
            if not all(isinstance(item, str) for item in results):
                continue

            expansion_list += results
        
        query_str = query.get_description()
        if self.retriever_type == "sparse":
            expansion_list = [query_str] + expansion_list 
            ' '.join(expansion_list)
        else:
            expansion_list = list(set([e.lower() for e in expansion_list]))
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)
        return expansions


class Q2D(QueryProcessor):
    cls_type = "q2d"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense"):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        Q2D @ https://arxiv.org/pdf/2305.03653
        query2doc @ https://arxiv.org/pdf/2303.07678
        GQE @ https://dl.acm.org/doi/pdf/10.1145/3589335.3651945
        """

        prompt = """
        Query: {query}
        Given a user's travel cities recommendation query, write a passage that answer the query.
            - Provide your answers in valid JSON format with double quotes: {{"answer": "YOUR ANSWER"}}.
        """

        cities = "\n".join(CITY_LIST)
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", cities=cities)},
            {"role": "assistant", "content": answer.format(answer="When planning a family-friendly vacation, choosing the right city can make all the difference in ensuring a memorable and enjoyable experience for everyone. Orlando, Florida, USA, known as the theme park capital of the world, offers attractions like Walt Disney World Resort, Universal Studios, and the Kennedy Space Center. San Diego, California, USA, is home to beautiful beaches, the world-famous San Diego Zoo, and Legoland California. Tokyo, Japan, offers Tokyo Disneyland and DisneySea, along with historic temples and delicious Japanese cuisine. These cities provide a range of attractions and activities designed to entertain and educate family members of all ages, ensuring a vacation filled with fun, learning, and unforgettable memories.", cities=cities)},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", cities=cities)},
            {"role": "assistant", "content": answer.format(answer="For photography enthusiasts seeking picturesque cities, there are several destinations around the world that offer breathtaking scenery and iconic landmarks. Paris, France, is a must-visit with its iconic Eiffel Tower, charming Montmartre district, and the historic Notre-Dame Cathedral. Venice, Italy, enchants photographers with its romantic canals, St. Mark's Basilica, and the vibrant colors of Burano Island. Kyoto, Japan, offers a serene setting with its traditional temples, Arashiyama Bamboo Grove, and the stunning Fushimi Inari Shrine. These cities provide an array of photogenic scenes that inspire creativity and capture the essence of each unique location, making them ideal destinations for photography enthusiasts.", cities=cities)},          
            {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", cities=cities)},
            {"role": "assistant", "content": answer.format(answer="For those who love horseback riding, certain cities around the world offer unique and scenic opportunities to explore their surroundings on horseback. Jackson, Wyoming, USA, is renowned for its breathtaking views of the Teton Range and offers horseback rides through Yellowstone and Grand Teton National Parks. Mendoza, Argentina, provides riders with the chance to explore the Andes Mountains and lush vineyards, offering stunning views and adventurous trails.  Lastly, the Scottish Highlands in the United Kingdom provide riders with rugged, dramatic landscapes, offering trails that pass through ancient forests, along lochs, and up mountainous terrain. These cities offer diverse and scenic riding experiences that allow enthusiasts to connect with nature while enjoying the thrill of horseback exploration.", cities=cities)},          
            {"role": "user", "content": prompt.format(query=query.get_description(), cities=cities)},
        ]

        response = self.llm.generate(message)
        pseudo_doc = response

        query_str = query.get_description()

        if self.retriever_type == "sparse":
            expansion_list = [query_str] + [pseudo_doc]
            expansions = ' '.join(expansion_list)
        
        else:
            expansion_list = [query_str] + [pseudo_doc]
            expansions = '[SEP]'.join(expansion_list)
        
        return expansions


class EQR(QueryProcessor):
    cls_type = "eqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, retriever_type: str = "dense", k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir, retriever_type=retriever_type)
        self.k = k

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        """
        cities = "\n".join(CITY_LIST)
        answer = ANSWER_FORMAT
        
        # origin 

        prompt = """
        Query: {query}

        Given a user's travel cities recommendation query, do the following steps:
            - Break down the query into {k} distinct keywords related to the given user query aspect.
            - For each keywords, provide a one-sentence description that clearly elaborates on its specific focus and relevance.
            - For each keywords, choose reonable example cities and concatenate them at the end of your one sentence description. 
            - Provide your answers in valid JSON format with double quotes: {{"answer": [LIST]}}, where keyword list is a list of string with description and example answers concatenate together.

        EXAMPLE QUERY: Picturesque cities for photography enthusiasts
        EXAMPLE LIST: ["Coastal Views - Captures the dynamic interface where the sea meets the land, offering picturesque views of beaches, cliffs, and marine life, such as Cape Town, Santorini”, "Natural Landscape Photography - Dedicated to capturing the untouched beauty of natural environments, focusing on the authenticity and raw elements of scenes, such as Queenstown, Faroe Islands, Fiji.", "Exotic Island Views - Highlights the distinctive beauty of island landscapes, featuring tropical beaches, lush vegetation, and serene tranquility, such as Phuket, Maldives, Galapagos Islands.", "Architectural Photography - Focuses on the artistic capture of buildings and architectural elements, emphasizing design, structure, and contextual interaction, such as Florence, Barcelona, Prague.", "Skyline Photography - Concentrates on capturing the iconic cityscapes and urban profiles from strategic vantage points, showcasing the layout and vibrancy of metropolitan areas, such as New York City, Hong Kong, Dubai.”]

        EXAMPLE QUERY: Family-Friendly Cities for Vacations
        EXAMPLE LIST: ["Theme Parks - Cities with expansive theme parks offering thrilling rides and attractions suitable for all ages such as Orlando and Los Angeles.", "Zoos and Aquariums - Feature diverse collections of animals and underwater displays such as Chicago and San Diego.", "Children Museums - Tailored for younger visitors with hands-on learning exhibits such as Indianapolis and New York City.", "Beaches - Safe, clean beaches with gentle waves and family amenities such as Honolulu and Miami.", "Parks - Large parks with playgrounds, picnic areas, and public events such as London and Vancouver."]
        """
        
        answer = ANSWER_FORMAT
        cities = ",".join(CITY_LIST)

        query_str = query.get_description()
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=5, cities=cities)},
            {"role": "assistant", "content": answer.format(answer=["Theme Parks - Cities with expansive theme parks offering thrilling rides and attractions suitable for all ages such as Orlando and Los Angeles.", "Zoos and Aquariums - Feature diverse collections of animals and underwater displays such as Chicago and San Diego.", "Children Museums - Tailored for younger visitors with hands-on learning exhibits such as Indianapolis and New York City.", "Beaches - Safe, clean beaches with gentle waves and family amenities such as Honolulu and Miami.", "Parks - Large parks with playgrounds, picnic areas, and public events such as London and Vancouver."])},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", k=5, cities=cities)},
            {"role": "assistant", "content": answer.format(answer=["Coastal Views - Captures the dynamic interface where the sea meets the land, offering picturesque views of beaches, cliffs, and marine life, such as Cape Town, Santorini", "Natural Landscape Photography - Dedicated to capturing the untouched beauty of natural environments, focusing on the authenticity and raw elements of scenes, such as Queenstown, Faroe Islands, Fiji.", "Exotic Island Views - Highlights the distinctive beauty of island landscapes, featuring tropical beaches, lush vegetation, and serene tranquility, such as Phuket, Maldives, Galapagos Islands.", "Architectural Photography - Focuses on the artistic capture of buildings and architectural elements, emphasizing design, structure, and contextual interaction, such as Florence, Barcelona, Prague.", "Skyline Photography - Concentrates on capturing the iconic cityscapes and urban profiles from strategic vantage points, showcasing the layout and vibrancy of metropolitan areas, such as New York City, Hong Kong, Dubai."])},
            {"role": "user", "content": prompt.format(query=query_str, k=self.k, cities=cities)},
        ]

        response = self.llm.generate(message)
        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        expansion_list = json.loads(response)["answer"]
        
        if self.retriever_type == "sparse":
            # append original aspect description
            query_list = [query_str]
            expansion_list = query_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)

        return expansions