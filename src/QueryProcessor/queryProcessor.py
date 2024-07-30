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
            # # determine whether the query is a broad query
            # is_broad = self._classify_query(query=query)

            # if is_broad:
            #     curr_query = Broad(description=query)
            # else: # activity
            #     curr_query = Activity(description=query)

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
    
    
    # def _classify_query(self, query: str) -> bool:
    #     """
    #     Determine the given query is a broad interest query.
    #     """
    #     prompt = """
    #     Here's a classification problem for a travel recommendation system:
    #     Given a query, please classify it into two classification, and inform the answer directly:
    #         1. Broad: These queries are very subjective and contains words that are very broad and wide. Examples: cities famous for [big and broad nouns], cities with [a sense of feeling], cities for [broad purpose], cities for [certain kind of people], etc.
    #         2. Activity: These queries are objective, and has very specific activities. Examples: Cities with [specific entities], Cities for [specific activities], etc.

    #     Requirements:
    #     - Your answer can only be "broad" or "activity"
    #     - Provide your answers strictly in JSON format: {{"answer": "YOUR ANSWER"}}


    #     Query:{query}
    #     """
    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are an expert in classification."},
    #         {"role": "user", "content": prompt.format(query=query)},
    #     ]

    #     response = self.llm.generate(message)
        
    #     try:
    #         start, end = response.find("{"), response.rfind("}") + 1
    #         answer = json.loads(response[start:end])["answer"]


    #         if answer == 'activity':
    #             return False
    #         return True

    #     except json.JSONDecodeError as e:
    #         print(f"Failed to parse JSON from response")
    #         print("GPT response: ", response)
    #         raise e

    #     except Exception as e:
    #         print(f"Failed to extract")
    #         print("GPT response: ", response)
    #         raise e
    

class GQR(QueryProcessor):
    cls_type = "gqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, k: int = 1):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
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
            query_list = [query_str] * 3
            paraphrase_list = query_list + paraphrase_list
            paraphrases = ' '.join(paraphrase_list)
        
        else: # for dense retriever
            paraphrase_list = [query_str] + paraphrase_list
            paraphrases = '[SEP]'.join(paraphrase_list)

        return paraphrases


class Q2E(QueryProcessor):
    cls_type = "q2e"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
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
            query_list = [query_str] * 5
            expansion_list = query_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)

        return expansions
    


class Q2A(QueryProcessor):
    cls_type = "q2a"
    def __init__(self, input_path: str, llm: LLM, output_dir: str):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
    
    def reformulate_query(self, query: AbstractQuery) -> str:
        if isinstance(query, Broad):
            new_desc = self.q2a(query=query.get_description(), k=10)
        else: # activity
            new_desc = self.q2a(query=query.get_description(), k=3)
        return new_desc

    def q2a(self, query: str, k: int) -> str:
        """
        """
        prompt = """
        Given a user's travel cities recommendation query, give a list of {k} activities for the given travel query.
        Provide your answers in valid JSON format with double quote: {{"answer": [LIST]}}, where keyword list is a list of string.

        query: {query}
        """
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=10)},
            {"role": "assistant", "content": answer.format(answer=["Visit interactive science museums", "Explore family-friendly parks", "Join city tours for families", "Attend family concerts", "Enjoy kid-friendly restaurants", "Visit zoos and aquariums", "Participate in art workshops", "Explore libraries for storytime", "Take family bike tours", "Visit amusement parks"])},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", k=10)},
            {"role": "assistant", "content": answer.format(answer=["Photograph sunsets and sunrises over landscapes", "Capture reflections in city lakes and rivers", "Explore botanical gardens for floral photography", "Shoot panoramic views from city outskirts", "Capture wildlife in urban nature reserves", "Photograph natural landmarks within the city", "Explore scenic trails for nature shots", "Capture seasonal changes in city parks", "Attend outdoor photography retreats", "Photograph starry nights from accessible city points"])},          
            {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", k=3)},
            {"role": "assistant", "content": answer.format(answer=["Explore guided trail rides in scenic areas", "Explore horseback mountain tours", "Join horseback riding beach tours"])},          
            {"role": "user", "content": prompt.format(query=query, k=k)},
        ]

        response = self.llm.generate(message)
        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        expansion_list = json.loads(response)["answer"]

        if self.retriever_type == "sparse":
            # append original query description
            query_list = [query] * 5
            expansion_list = query_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query] + expansion_list
            expansions = '[SEP]'.join(expansion_list)

        return expansions



class GenQREnsemble(QueryProcessor):
    cls_type = "genqr"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, n: int = 5, k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
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
            expansion_list = [query_str] * 5 + expansion_list 
            ' '.join(expansion_list)
        else:
            expansion_list = list(set([e.lower() for e in expansion_list]))
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)
        return expansions


class Q2D(QueryProcessor):
    cls_type = "q2d"
    def __init__(self, input_path: str, llm: LLM, output_dir: str, k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
        self.k = k

    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        Q2D @ https://arxiv.org/pdf/2305.03653
        query2doc @ https://arxiv.org/pdf/2303.07678
        GQE @ https://dl.acm.org/doi/pdf/10.1145/3589335.3651945
        """
        prompt = """
        Query: {query}
        Given a user's travel cities recommendation query, write a passage that answer the query by providing {k} cities recommendation from city list.
            - Provide one-sentence rationale and attractions for each recommendation.
            - Manually check whether your example cities come from the choice of cities, if not, replace with a new one in the list.
            - Provide your answers in valid JSON format with double quotes: {{"answer": "YOUR ANSWER"}}.
        
        This is your choice of cities:
        {cities}
        """
        answer = ANSWER_FORMAT
        cities = ",".join(CITY_LIST)

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Family-Friendly Cities for Vacations", k=5, cities=cities)},
            {"role": "assistant", "content": answer.format(answer="Orlando, Florida is a prime destination for families, boasting world-renowned theme parks like Disney World and Universal Studios. San Diego, California offers a relaxed vibe with family-friendly attractions such as the San Diego Zoo and beautiful beaches. Kyoto, Japan provides a culturally enriching experience with its historic sites and child-friendly museums. Copenhagen, Denmark is perfect for families, featuring the enchanting Tivoli Gardens and numerous other kid-friendly activities. Queenstown, New Zealand caters to adventure-loving families with its stunning landscapes and outdoor activities suitable for all ages.")},
            {"role": "user", "content": prompt.format(query="Picturesque cities for photography enthusiasts", k=5, cities=cities)},
            {"role": "assistant", "content": answer.format(answer="Venice, Italy captivates photography enthusiasts with its iconic waterways and historic architecture. Kyoto, Japan offers stunning photo opportunities with its well-preserved temples and beautiful cherry blossoms in spring. New York City, USA is a photographer's playground with its vibrant street life, towering skyscrapers, and iconic landmarks like Times Square and Central Park. Cape Town, South Africa boasts breathtaking landscapes from Table Mountain to picturesque beaches. Reykjavik, Iceland provides unique photographic scenes with its dramatic volcanic landscapes and opportunities to capture the Northern Lights.")},          
            {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", k=3, cities=cities)},
            {"role": "assistant", "content": answer.format(answer="Calgary, Canada is renowned for its annual Calgary Stampede, offering abundant horseback riding opportunities in scenic Alberta. Lexington, Kentucky, known as the 'Horse Capital of the World', features vast horse farms and equestrian events. Cordoba, Argentina offers a rich history of horseback riding with traditional estancias where visitors can experience gaucho culture. ")},          
            {"role": "user", "content": prompt.format(query=query.get_description(), k=self.k, cities=cities)},
        ]

        response = self.llm.generate(message)
        pseudo_doc = response

        query_str = query.get_description()

        if self.retriever_type == "sparse":
            expansion_list = [query_str] * 5 + [pseudo_doc]
            expansions = ' '.join(expansion_list)
        
        else:
            expansion_list = [query_str] + [pseudo_doc]
            expansions = '[SEP]'.join(expansion_list)
        
        return expansions


class EQR(QueryProcessor):
    cls_type = "eqr"
    def __init__(self, input_path: str, llm: LLM,output_dir: str, k: int = 5):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param output_dir:
        """ 
        super().__init__(input_path=input_path, llm=llm, output_dir=output_dir)
        self.k = k


    def reformulate_query(self, query: AbstractQuery) -> str:
        """
        """
        prompt = """
        Query: {query}

        Given a user's travel cities recommendation query, do the following steps:
            - Break down the query into {k} distinct keywords related to the given user query aspect.
            - For each keywords, provide a one-sentence description that clearly elaborates on its specific focus and relevance.
            - For each keywords, choose 3 example cities and concatenate them at the end of your one sentence description. 
            - Manually check whether your example cities come from the choice of cities, if not, replace with a new one in the list.
            - Provide your answers in valid JSON format with double quotes: {{"answer": [LIST]}}, where keyword list is a list of string with description and example answers concatenate together.
        
        This is your choice of cities:
        {cities}

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
            query_list = [query_str] * 5
            expansion_list = query_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query_str] + expansion_list
            expansions = '[SEP]'.join(expansion_list)

        return expansions


    # def process_activity_query(self, query: str) -> str:
    #     """
    #     """
    #     prompt = """
    #     Query: {query}

    #     Given a user's travel cities recommendation query, do the following steps:
    #         - Provide a one or two sentence description with various keywords related to the query.
    #         - Choose 10 best cities that satisfy the query and concatenate them at the end of your one sentence description.  
    #         3. Provide your answers in valid JSON format with double quotes: {{"answer": RESULT}}, where RESULT is the concatenated description and example answers.

    #     EXAMPLE QUERY: Cities popular for horseback riding
    #     EXAMPLE RESULT: Cities renowned for excellent horseback riding opportunities often feature stunning natural landscapes suitable for horse riding such as mountain, forest, or beaches, well-maintained horse riding trails, and a rich cultural heritage related to horse riding. Some of the best cities for this activity include Lexington (Kentucky), Mendoza, Queenstown (New Zealand), Mont-Tremblant, Cape Town, Cabo San Lucas, Santa Fe, Victoria Falls, and Asheville.

    #     This is your choice of cities:
    #     {cities}
    #     """
    #     answer = ANSWER_FORMAT
    #     cities = ",".join(CITY_LIST)

    #     message = [
    #         {"role": "system", "content": "You are a travel expert."},
    #         {"role": "user", "content": prompt.format(query="Cities popular for horseback riding", cities=cities)},
    #         {"role": "assistant", "content": answer.format(answer="Cities renowned for excellent horseback riding opportunities often feature stunning natural landscapes in mountain or forest, well-maintained horse riding trails, beautiful beach and coastlines, and a rich cultural heritage related to horse riding. Some of the best cities for this activity include Lexington (Kentucky), Mendoza, Queenstown (New Zealand), Mont-Tremblant, Cape Town, Cabo San Lucas, Santa Fe, Victoria Falls, and Asheville.")},        
    #         {"role": "user", "content": prompt.format(query=query, cities=cities)},
    #     ]

    #     response = self.llm.generate(message)

    #     if self.retriever_type == "sparse":
    #         expansion_list = [query] * 5 + [response]
    #         expansions = ' '.join(expansion_list)
        
    #     else:
    #         expansions = f"{query}: {response}"
        
    #     return expansions



############################################
############# Tree method ##################
# class Tree(QueryProcessor):
#     def __init__(self, input_path: str, llm: LLM, retriever_type: str, output_dir: str, max_subtopics: int = 5, depth_limit: int = 3):
#         """
#         Initialize the query processor
#         :param query:
#         :param llm:
#         :param output_dir:
#         """ 
#         super().__init__(input_path=input_path, llm=llm, retriever_type=retriever_type, output_dir=output_dir)
#         self.max_subtopics = max_subtopics
#         self.depth_limit = depth_limit


#     def elaborate_tree(self, query: str) -> str:
#         """
#         """
#         expandable = self.is_expandable(query=query)
#         print(f"{query}: {expandable}")
#         if (not expandable) and self.max_subtopics <= 0: # base case
#             return query
        
#         else: # recursive case
#             subtopics = self.expand_subtopics(query=query, max_subtopics=self.max_subtopics)
#             subtopics_expansion = [self.elaborate_tree(query=sub, max_subtopics=self.max_subtopics, depth_limit=self.depth_limit - 1) for sub in subtopics]
#             return '\n'.join(subtopics_expansion)


#     def expand_subtopics(self, query: str, max_subtopics: int) -> list[str]:
#         """
#         """
#         prompt = """
#         Given the provided query about travel destination cities recommendation, expand the query to up to {max_subtopics} subtopics from different aspects that represent certain attributes a city might have.
#             1. Each subtopic should represent a more detailed granularity of the travel concept than the original topic.
#             2. Each subtopic should be mutually exclusive and focus on a different travel-related concept.
#             3. These subtopics will form new queries about travel destination cities recommendation, so provide them in a way that can be answered by a list of cities. Write them concisely in very short sentences.
#             4. You do not need to, and not expected to, exhaust the maximum number of subtopics; {max_subtopics} is merely an upper limit.
#             5. Provide your answers in valid JSON format with double quotes: {{"answer": [SUBTOPICS LIST]}}.

#         Provide your answers in valid JSON format with double quotes: {{"answer": [SUBTOPICS LIST]}}.

#         query: {query}
#         """
#         answer = ANSWER_FORMAT

#         message = [
#             {"role": "system", "content": "You are a travel expert."},
#             {"role": "user", "content": prompt.format(query="Family-friendly cities.")},
#             {"role": "assistant", "content": answer.format(answer=["Cities with educational museums.", "Cities with best-amusement parks.", "Cities with top-rated family hotels."])},
#             {"role": "user", "content": prompt.format(query="Good food cities.")},
#             {"role": "assistant", "content": answer.format(answer=["Cities with street food.", "Cities with Michelin-starred restaurants.", "Cities famous for specific local cuisines."])},
#             {"role": "user", "content": prompt.format(query=query, max_subtopics=max_subtopics)},
#         ]

#         response = self.llm.generate(message)

#         # parse the answer
#         start = response.find("{")
#         end = response.rfind("}") + 1
#         response = response[start:end]
#         subtopics = json.loads(response)["answer"]
#         return subtopics
    

#     def is_expandable(self, query: str) -> bool:
#         """
#         """
#         prompt = """
#         Subtopics mining is a strategy used in information retrieval systems to identify related, more specific subtopics under a given broad query, which helps uncover additional topics that may interest users. Each subtopic should be significantly more specific in terminology and coverage than the original query topics and should be very distinct from one another.

#         Consider a travel destination recommender system where the goal is to suggest a list of cities based on user queries. We employ subtopics mining to better interpret and respond to these queries. Our primary data source is Wiki-Travel, which often lacks detailed information about specific cities. Therefore, it is crucial to maintain a balance in the level of granularity when mining subtopics.

#         Conversely, here are some additional examples where the query might point to an activity that is too specific to mine meaningful subtopics and/or find relevant information from Wiki-Travel (please see the above examples without further brackets as additional negative cases):

#         * Cities famous for food [Cities known for special local cuisines [Cities famous for Thai-style cuisine; Cities famous for Pizza …]; Cities with numerous high-end Michelin star restaurants; Cities known for vibrant street food cultures.]
#         * Cities rich in entertainment [Cities with famous music scenes; Cities known for Broadway-style productions; Cities with bustling nightclubs and bars; Cities with good amusement park [Cities with good theme park …]].
#         * Family-friendly cities [Cities with kid-focused museums and zoos [Cities with fun art galleries; Cities with educational natural history museum …]; Cities with large recreational parks and fun zones [Cities with good theme park; Cities with fun water parks …]].
#         * Cities for sports lovers [Cities with major league teams and stadiums; Cities ideal for outdoor advantures [Cities famous for mountain climbing …]; Cities for winter sports lover [Cities famous for skiing …]; Football cities].
#         * Budget-friendly cities [Cities with budget lodging options; Cities with numerous free tourist attractions; Cities offering great food at low prices; Cities with cheap transportation and local facilities].
#         * Cities with a romantic atmosphere [Cities with cozy, romantic restaurants; Cities offering romantic activities [ Cities offering hot air balloon rides …]].
#         * Luxury cities [Cities with high-end shopping districts; Cities with exclusive, gourmet restaurants; Cities featuring luxury hotels and resorts].

#         Conversely, here are some additional examples where the query might point to a activity that is too specific to mine meaningful subtopics and/or finding relevant information from Wiki-Travel (please see the above example without further bracket as additional negative cases):

#         * Cities famous for street food or any kind of specific food genre [too specific to find relevant info from wiki for further mining].
#         * Cities with good theme park [too specific to find relevant info from wiki for further mining].
#         * Cities famous for their educational natural science museums [too specific for further mining].
#         * Best cities to watch NBA games [too specific for further mining].
#         * Cities with budget lodging options [too specific to find relevant info from wiki for further mining].
#         * Cities with cozy, romantic restaurants [too specific to find relevant info from wiki for further mining].
#         * Cities with luxury hotels [too specific to find relevant info from wiki for further mining].

#         Given the following information and the broadness of the Wiki-Travel data source (be cautious, as many details cannot be found in Wiki-Travel), decide whether the given query is suitable for further subtopics mining (i.e., further expanding it into subtopics).

#         Provide your answers in valid JSON format with double quotes: {{"answer": "yes" | "no"}}. 

#         Query: {query}
#         """
#         # answer = ANSWER_FORMAT

#         message = [
#             {"role": "system", "content": "You are a travel expert."},
#             {"role": "user", "content": prompt.format(query=query)},
#         ]

#         response = self.llm.generate(message).lower()
#         if "yes" in response:
#             return True
#         elif "no" in response:
#             return False
#         else:
#             raise ValueError("")