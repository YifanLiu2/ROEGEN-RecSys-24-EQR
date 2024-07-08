import json, os, re, pickle
from tqdm import tqdm

from src.LLM.GPTChatCompletion import *
from src.Entity.query import *
from src.Entity.aspect import *

ANSWER_FORMAT = """{{"answer": {answer}}}"""
RETRIEVER_TYPE = {"dense", "sparse"}

class queryProcessor:
    """
    Query Processor class
    """
    def __init__(self, input_path: str, llm: LLM, mode_name: str = None, retriever_type: str = None, output_dir: str = "output"):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param mode_name: can only be "gqr", "q2e", "q2d", "genqr", "elaborate", "answer",
        :param output_dir:
        """ 
        self.llm = llm
        self.mode_name = mode_name
        self.output_dir = output_dir
        self.query_list = self._load_queries(input_path)
        self.retriever_type = retriever_type
    
    def process_query(self) -> list[Query]:
        """
        Process the queries
        """
        # fetch aspect processing function
        aspect_processing_func = None
        if self.mode_name == "gqr":
            aspect_processing_func = self._GQR
        elif self.mode_name == "q2e":
            aspect_processing_func = self._Q2E
        elif self.mode_name == "q2d":
            aspect_processing_func = self._Q2D
        elif self.mode_name == "genqr":
            aspect_processing_func = self._GenQREnsemble
        elif self.mode_name == "elaborate":
            aspect_processing_func = self._elaborate_aspect
        elif self.mode_name == "answer":
            aspect_processing_func = self._answer_aspect
        
        

        result_queries = []
        for query in tqdm(self.query_list, desc="Processing queries", unit="query"):
            curr_query = Query(description=query)

            # extract preferences and constraints
            preferences, constraints, hybrids = self._extract_aspects(query=query)

            # preferences
            for p in preferences:
                preference = Preference(description=p)
                curr_query.add_preference(preference)

            # constraints
            for c in constraints:
                constraint = Constraint(description=c)
                curr_query.add_constraint(constraint)

            # hybrids
            for h in hybrids:
                hybrid = Hybrid(description=h)
                curr_query.add_hybrid(hybrid)

            if aspect_processing_func is not None:
                for aspect in curr_query.get_all_aspects():
                    new_desc = aspect_processing_func(query_aspect=aspect.description)
                    aspect.set_new_description(new_description=new_desc)
            
            result_queries.append(curr_query)

        self._save_results(result=result_queries)
        return result_queries
    
    def _load_queries(self, input_path: str) -> list[str]:
        """
        """
        with open(input_path, "r") as f:
            queries = [line.strip() for line in f]
        return queries
    
    def _save_results(self, result: list[Query]):
        """
        """
        pkl_path = os.path.join(self.output_dir, f"processed_query_{self.mode_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

        query_data = []
        for query in result:
            query_info = {
                "Query": query.get_description(),
                "Preferences": [
                    {
                        preference.get_original_description(): preference.get_new_description()
                    } for preference in query.get_preferences()
                ],
                "Constraints": [
                    {
                        constraint.get_original_description(): constraint.get_new_description()
                    } for constraint in query.get_constraints()
                ],
                "Hybrids": [
                    {
                        hybrid.get_original_description(): hybrid.get_new_description()
                    } for hybrid in query.get_hybrids()
                ]
            }
            query_data.append(query_info)

        json_path = os.path.join(self.output_dir, f"processed_query_{self.mode_name}.json")
        with open(json_path, 'w') as f:
            json.dump(query_data, f, indent=4)
        
    def _extract_aspects(self, query: str) -> tuple[list[str], list[str], list[str]]:
        """
        Extract preferences and constraints from the query
        """
        # actual prompt
        # Corrected prompt string
        prompt = """
        Given a user's query about travel destination city recommendation, please categorize each aspect of the query into the following types:

        1. "Constraint" - This category includes highly specific aspects that target unique entities or activities. Only those aspects that are clearly defined and rare fit into this category.

        2. "Activities" - This category includes all other broader entities or activities. broader aspects that describe general types of activities or entities. It includes anything not specific enough to be a constraint.

        3. "Preferences" - This category includes subjective preferences that are not directly related to specific entities or activities.

        Chain of Thought Process:
        - Determine if the aspect pertains to specific entities, activities, or is a subjective preference.
        - If the aspect is not related to specific entities or activities, it is a preference. If it is a very specific and rare entity or activity, it is a constraint. Otherwise, it is an activity.

        Requirements:
        - Each aspect should be expressed succinctly to ensure clarity and precision in the recommendation process.
        - Provide your answers strictly in JSON format: {{"answer": {{"constraint": [], "preferences": [], "activities": []}}}}

        Query: {query}
        """

        # define answer format
        answer = ANSWER_FORMAT

        # few-shots
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Recommend me cities with historical sites and museums to explore during my travels?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraint": [],"preferences": [],"activities": ["cities with historical sites", "cities with museums"]}}))},
            {"role": "user", "content": prompt.format(query="I am plannning on a trip to cities with good restaurants. ")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraint": [],"preferences": [],"activities": ["cities with good restaurants"]}}))},
            {"role": "user", "content": prompt.format(query="I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there? ")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraint": ["in Asia"], "preferences": ["cities on a budget"], "activities": []}}))},
            {"role": "user", "content": prompt.format(query="I want a city with a major film festival in June and good seafood restaurants.")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraint": [], "preferences": [], "activities": ["cities with good seafood restaurants", "cities with major film festival in June"]}}))},
            {"role": "user", "content": prompt.format(query=query)},
        ]

        response = self.llm.generate(message)
        
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            answer = json.loads(response[start:end])["answer"]
            preferences = answer["preferences"]
            constraints = answer["constraint"]
            hybrids = answer["activities"]
            return preferences, constraints, hybrids

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print(f"Failed to extract constraints")
            print("GPT response: ", response)
            raise e
        

    # def _answer_constraints(self, query_constraint: str) -> str:
    #     """
    #     Answer the constraints
    #     """
    #     prompt = """
    #     Given the specified constraints of a user's travel cities recommendation query, generate a list of all cities that meet the constraint.
    #     Provide your answers in valid JSON format with double quote: {{"answer": "Cities"}}.

    #     Constraints: {constraint}
    #     """

    #     # define answer format
    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a travel expert."},
    #         {"role": "user", "content": prompt.format(constraint="Universal Studios")},
    #         {"role": "assistant", "content": answer.format(answer="Los Angeles, Orlando, Singapore, Osaka, Beijing")},  
    #         {"role": "user", "content": prompt.format(constraint="Disney")},
    #         {"role": "assistant", "content": answer.format(answer="Anaheim, Orlando, Tokyo, Paris, Shanghai, Hong Kong")},  
    #         {"role": "user", "content": prompt.format(constraint=query_constraint)},
    #     ]

    #     response = self.llm.generate(message)

    #     # parse response
    #     return correct_and_extract(response) + f", {query_constraint}" * 5


    # def _reformulate_aspect(self, query_aspect: str) -> str:
    #     """
    #     Reformulate one aspect of the query
    #     """
    #     prompt = """
    #     Given the specified aspect of a user's travel cities recommendation query, generate one sentence reformulation of the aspect with reasonable user intent to better reflect the user's intent. 
    #     Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        
    #     Aspect: {aspect}
    #     """

    #     # define answer format
    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a travel expert."},
    #         {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
    #         {"role": "assistant", "content": answer.format(answer="Which cities are best suited for adventure seekers looking for thrilling activities?")},
    #         {"role": "user", "content": prompt.format(aspect="has historical sites")},
    #         {"role": "assistant", "content": answer.format(answer="Which cities are rich in historical sites and cultural heritage?")},          
    #         {"role": "user", "content": prompt.format(aspect="on Asia")},
    #         {"role": "assistant", "content": answer.format(answer="Which cities in Asia are recommended for travelers?")},  
    #         {"role": "user", "content": prompt.format(aspect=query_aspect)},
    #     ]
        
    #     response = self.llm.generate(message)

    #     # parse response
    #     return correct_and_extract(response)
        

    # def _expand_aspect(self, query_aspect: str) -> str:
    #     """
    #     Expand one aspect of the query
    #     """
    #     prompt = """
    #     Given the specified aspect of a user's travel cities recommendation query, generate a list of three similar but improved descriptions of the aspect to better reflect the user's intent. 
    #     Provide your answers in valid JSON format with double quote: {{"answer": []}}.

    #     Aspect: {aspect}
    #     """
    #     # define answer format
    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a travel expert."},
    #         {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
    #         {"role": "assistant", "content": answer.format(answer=["Ideal for outdoor adventure enthusiasts.", "Great for those seeking thrilling adventure activities.", "Perfect for adventure seekers looking for excitement and challenges."])},
    #         {"role": "user", "content": prompt.format(aspect="has historical sites")},
    #         {"role": "assistant", "content": answer.format(answer=["Rich in historical landmarks and museums.", "Offers a wealth of cultural and historical sites.", "Filled with historical attractions and heritage sites."])},          
    #         {"role": "user", "content": prompt.format(aspect="on Asia")},
    #         {"role": "assistant", "content": answer.format(answer=["Located in Asia.", "Situated in Asia.", "In Asia."])},  
    #         {"role": "user", "content": prompt.format(aspect=query_aspect)},
    #     ]

    #     response = self.llm.generate(message)

    #      # parse response
         
    #      # find the answer part
    #     start = response.find("{")
    #     end = response.rfind("}") + 1
    #     response = response[start:end]
    #     expansion_list = json.loads(response)["answer"]
    #     expansion_list.append(query_aspect)

    #     # join the list into a string
    #     expansions = ' '.join(expansion_list)
        
    #     return expansions

    def elaborate_tree(self, query_aspect: str, max_subtopics: int = 5, depth_limit: int = 3) -> str:
        """
        """
        expandable = self.is_expandable(query_aspect=query_aspect)
        print(f"{query_aspect}: {expandable}")
        if (not expandable) and depth_limit <= 0: # base case
            return query_aspect
        
        else: # recursive case
            subtopics = self.expand_subtopics(query_aspect=query_aspect, max_subtopics=max_subtopics)
            print(f"Aspect: {query_aspect}")
            print(f"Subtopics: {subtopics}")
            subtopics_expansion = [self.elaborate_tree(query_aspect=sub, max_subtopics=max_subtopics, depth_limit=depth_limit - 1) for sub in subtopics]
            return '\n'.join(subtopics_expansion)


    def expand_subtopics(self, query_aspect: str, max_subtopics: int) -> list[str]:
        """
        """
        prompt = """
        Given the provided query about travel destination cities recommendation, expand the query to up to {max_subtopics} subtopics from different aspects that represent certain attributes a city might have.
            1. Each subtopic should represent a more detailed granularity of the travel concept than the original topic.
            2. Each subtopic should be mutually exclusive and focus on a different travel-related concept.
            3. These subtopics will form new queries about travel destination cities recommendation, so provide them in a way that can be answered by a list of cities. Write them concisely in very short sentences.
            4. You do not need to, and not expected to, exhaust the maximum number of subtopics; {max_subtopics} is merely an upper limit.
            5. Provide your answers in valid JSON format with double quotes: {{"answer": [SUBTOPICS LIST]}}.

        Provide your answers in valid JSON format with double quotes: {{"answer": [SUBTOPICS LIST]}}.

        Aspect: {aspect}
        """
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="Family-friendly cities.")},
            {"role": "assistant", "content": answer.format(answer=["Cities with educational museums.", "Cities with best-amusement parks.", "Cities with top-rated family hotels."])},
            {"role": "user", "content": prompt.format(aspect="Good food cities.")},
            {"role": "assistant", "content": answer.format(answer=["Cities with street food.", "Cities with Michelin-starred restaurants.", "Cities famous for specific local cuisines."])},
            {"role": "user", "content": prompt.format(aspect=query_aspect, max_subtopics=max_subtopics)},
        ]

        response = self.llm.generate(message)

        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        subtopics = json.loads(response)["answer"]
        return subtopics
    

    def is_expandable(self, query_aspect: str) -> bool:
        """
        """
        prompt = """
        Subtopics mining is a strategy used in information retrieval systems to identify related, more specific subtopics under a given broad query, which helps uncover additional topics that may interest users. Each subtopic should be significantly more specific in terminology and coverage than the original query topics and should be very distinct from one another.

        Consider a travel destination recommender system where the goal is to suggest a list of cities based on user queries. We employ subtopics mining to better interpret and respond to these queries. Our primary data source is Wiki-Travel, which often lacks detailed information about specific cities. Therefore, it is crucial to maintain a balance in the level of granularity when mining subtopics.

        Conversely, here are some additional examples where the query might point to an activity that is too specific to mine meaningful subtopics and/or find relevant information from Wiki-Travel (please see the above examples without further brackets as additional negative cases):

        * Cities famous for food [Cities known for special local cuisines [Cities famous for Thai-style cuisine; Cities famous for Pizza …]; Cities with numerous high-end Michelin star restaurants; Cities known for vibrant street food cultures.]
        * Cities rich in entertainment [Cities with famous music scenes; Cities known for Broadway-style productions; Cities with bustling nightclubs and bars; Cities with good amusement park [Cities with good theme park …]].
        * Family-friendly cities [Cities with kid-focused museums and zoos [Cities with fun art galleries; Cities with educational natural history museum …]; Cities with large recreational parks and fun zones [Cities with good theme park; Cities with fun water parks …]].
        * Cities for sports lovers [Cities with major league teams and stadiums; Cities ideal for outdoor advantures [Cities famous for mountain climbing …]; Cities for winter sports lover [Cities famous for skiing …]; Football cities].
        * Budget-friendly cities [Cities with budget lodging options; Cities with numerous free tourist attractions; Cities offering great food at low prices; Cities with cheap transportation and local facilities].
        * Cities with a romantic atmosphere [Cities with cozy, romantic restaurants; Cities offering romantic activities [ Cities offering hot air balloon rides …]].
        * Luxury cities [Cities with high-end shopping districts; Cities with exclusive, gourmet restaurants; Cities featuring luxury hotels and resorts].

        Conversely, here are some additional examples where the query might point to a activity that is too specific to mine meaningful subtopics and/or finding relevant information from Wiki-Travel (please see the above example without further bracket as additional negative cases):

        * Cities famous for street food or any kind of specific food genre [too specific to find relevant info from wiki for further mining].
        * Cities with good theme park [too specific to find relevant info from wiki for further mining].
        * Cities famous for their educational natural science museums [too specific for further mining].
        * Best cities to watch NBA games [too specific for further mining].
        * Cities with budget lodging options [too specific to find relevant info from wiki for further mining].
        * Cities with cozy, romantic restaurants [too specific to find relevant info from wiki for further mining].
        * Cities with luxury hotels [too specific to find relevant info from wiki for further mining].

        Given the following information and the broadness of the Wiki-Travel data source (be cautious, as many details cannot be found in Wiki-Travel), decide whether the given query is suitable for further subtopics mining (i.e., further expanding it into subtopics).

        Provide your answers in valid JSON format with double quotes: {{"answer": "yes" | "no"}}. 

        Query: {aspect}
        """
        # answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)
        response = correct_and_extract(response).lower()
        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            raise ValueError("")


    def _elaborate_aspect(self, query_aspect: str) -> str:
        """
        Elaborate one aspect of the query
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate one paragraph with specific definition of the aspect. You could focus on taxonomizing the different sub-aspects or providing in-depth information about the aspect. 
        Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        
        Aspect: {aspect}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer="Cities that are suitable for adventure seekers are those that offer a variety of thrilling and exciting activities tailored to those who crave physical challenges and adrenaline-pumping experiences. These cities often feature natural landscapes that allow for activities like hiking, climbing, white-water rafting, and skydiving. Additionally, they might host adventure parks or offer unique local experiences such as jungle expeditions or desert safaris. Such destinations are designed to cater to the adventurous spirit, providing not just activities but also the necessary safety measures and facilities to ensure a memorable and exhilarating visit.")},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer="Cities that are noted for having historical sites are rich in monuments, ruins, and museums that chronicle significant past events and cultures. These cities serve as gate-banners of history, often featuring well-preserved architecture, ancient artifacts, and UNESCO World Heritage sites that attract scholars, history enthusiasts, and tourists alike. The presence of these historical sites adds a deep cultural layer to the city, offering visitors a tangible connection to the past and an opportunity to learn about the historical narratives that shaped the modern world. Such cities often provide guided tours, educational programs, and interactive exhibits to enhance the visitor experience.")},          
            {"role": "user", "content": prompt.format(aspect="has Disney")},
            {"role": "assistant", "content": answer.format(answer= "Cities with Disney theme parks are renowned destinations that promise magical experiences for visitors of all ages. Anaheim, California, is home to Disneyland Resort, the original Disney theme park, offering classic attractions and the newer Star Wars: Galaxy’s Edge. Orlando, Florida, hosts Walt Disney World Resort, the largest Disney park globally, featuring four theme parks and two water parks. Tokyo, Japan, provides a unique twist with Tokyo Disney Resort, including Tokyo Disneyland and Tokyo DisneySea, known for its high attention to detail and cultural adaptations. Paris, France, offers Disneyland Paris, blending Disney magic with European flair. These cities not only draw Disney enthusiasts but also offer comprehensive family entertainment, making them top choices for Disney-themed vacations.")},  
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)

        # parse response
        return correct_and_extract(response)
    
    def _answer_aspect(self, query_aspect: str) -> str:
        """
        Elaborate one aspect of the query with exposed answer
        """
        prompt = """
        Given the specified constraints of a user's travel cities recommendation query, generate a comprehensive list of cities that meet the constraint and round small cities into the nearest large cities. Also, please provide a short justification for each city.
        Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        
        Aspect: {aspect}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer="Cities that are suitable for adventure seekers are those that offer a variety of thrilling and exciting activities tailored to those who crave physical challenges and adrenaline-pumping experiences. These cities often feature natural landscapes that allow for activities like hiking, climbing, white-water rafting, and skydiving. Additionally, they might host adventure parks or offer unique local experiences such as jungle expeditions or desert safaris. Such destinations are designed to cater to the adventurous spirit, providing not just activities but also the necessary safety measures and facilities to ensure a memorable and exhilarating visit.")},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer="Cities that are noted for having historical sites are rich in monuments, ruins, and museums that chronicle significant past events and cultures. These cities serve as gate-banners of history, often featuring well-preserved architecture, ancient artifacts, and UNESCO World Heritage sites that attract scholars, history enthusiasts, and tourists alike. The presence of these historical sites adds a deep cultural layer to the city, offering visitors a tangible connection to the past and an opportunity to learn about the historical narratives that shaped the modern world. Such cities often provide guided tours, educational programs, and interactive exhibits to enhance the visitor experience.")},          
            {"role": "user", "content": prompt.format(aspect="has Disney")},
            {"role": "assistant", "content": answer.format(answer= "Cities with Disney theme parks are renowned destinations that promise magical experiences for visitors of all ages. Anaheim, California, is home to Disneyland Resort, the original Disney theme park, offering classic attractions and the newer Star Wars: Galaxy’s Edge. Orlando, Florida, hosts Walt Disney World Resort, the largest Disney park globally, featuring four theme parks and two water parks. Tokyo, Japan, provides a unique twist with Tokyo Disney Resort, including Tokyo Disneyland and Tokyo DisneySea, known for its high attention to detail and cultural adaptations. Paris, France, offers Disneyland Paris, blending Disney magic with European flair. These cities not only draw Disney enthusiasts but also offer comprehensive family entertainment, making them top choices for Disney-themed vacations.")},  
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)

        # parse response
        return correct_and_extract(response)
    

    def _Q2D(self, query_aspect: str) -> str:
        """
        Q2D @ https://arxiv.org/pdf/2305.03653
        query2doc @ https://arxiv.org/pdf/2303.07678
        GQE @ https://dl.acm.org/doi/pdf/10.1145/3589335.3651945
        """
        prompt = """
        Given the specified constraints of a user's travel cities recommendation query, write a passage that answer the following query by providing some cities recommendation.
        Give the rational before answering.
        Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        
        Aspect: {aspect}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer="For travelers seeking thrilling adventures, several cities around the world stand out as prime destinations: 1. Queenstown, New Zealand: Often hailed as the 'Adventure Capital of the World,' Queenstown offers a stunning array of activities ranging from bungee jumping and skydiving to jet boating and mountain biking, all set against the dramatic backdrop of the Southern Alps. 2. Interlaken, Switzerland: Nestled between two lakes and surrounded by the Swiss Alps, Interlaken is a haven for adventure enthusiasts. Here, you can indulge in paragliding, canyoning, and white-water rafting, or take a more scenic approach with hiking and mountain climbing. 3. Moab, Utah, USA: Moab is the gateway to the rugged beauty of the red rock landscapes of Arches and Canyonlands National Parks. It's a top destination for rock climbing, mountain biking, and off-roading—perfect for those who seek adrenaline-fueled escapades. Each of these cities not only caters to thrill-seekers but also offers breathtaking natural beauty, providing a perfect blend of adventure and awe-inspiring vistas.")},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer="For those interested in exploring historical sites, several cities around the world are renowned for their rich history and cultural heritage: 1. Rome, Italy: Known as the 'Eternal City,' Rome is a treasure trove of antiquity, featuring iconic landmarks such as the Colosseum, Roman Forum, and the Pantheon, each telling tales of a bygone era. 2. Athens, Greece: The cradle of Western civilization, Athens is home to ancient marvels like the Acropolis and the Temple of Olympian Zeus, offering a deep dive into ancient Greek history. 3. Cairo, Egypt: Cairo provides a gateway to the past, where visitors can marvel at the wonders of Ancient Egypt, including the Great Pyramids of Giza and the Sphinx, alongside a rich tapestry of medieval architecture. Each of these cities not only preserves its historical legacy but also offers an immersive experience into the history that shaped our world today.")},          
            {"role": "user", "content": prompt.format(aspect="has Disney")},
            {"role": "assistant", "content": answer.format(answer= "For those who cherish the magical experience of Disney, certain cities around the world host Disney theme parks that promise fun-filled adventures for all ages: 1. Orlando, Florida, USA: Home to Walt Disney World Resort, Orlando offers a massive expanse of theme parks including Magic Kingdom, Epcot, Disney’s Hollywood Studios, and Disney’s Animal Kingdom. 2. Anaheim, California, USA: Anaheim is the site of Disneyland Park, the very first of the Disney parks, along with Disney California Adventure Park. It provides a historic look at the legacy of Disney while offering modern attractions. 3. Tokyo, Japan: Tokyo Disney Resort includes Tokyo Disneyland and Tokyo DisneySea, each offering unique attractions and shows with distinct themes that are beloved by visitors from all over the world. These cities not only bring the wonder of Disney to life but also offer a wide range of other entertainment options to ensure a memorable vacation for families, Disney enthusiasts, and casual visitors alike.")},  
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)

        # parse response
        pseudo_doc = correct_and_extract(response)

        if self.retriever_type == "sparse":
            expansion_list = [query_aspect] * 5 + [pseudo_doc]
            expansions = ' '.join(expansion_list)
        
        else:
            expansion_list = [query_aspect] + [pseudo_doc]
            expansions = '\n'.join(expansion_list)
        
        return expansions


    def _Q2E(self, query_aspect: str) -> str:
        """
        Q2E @ https://arxiv.org/pdf/2305.03653
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate a list of keywords related to the given user query aspect. 
        Provide your answers in valid JSON format with double quote: {{"answer": []}}.

        Aspect: {aspect}
        """
        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer=["adventure sports", "extreme sports", "hiking trails", "national parks", "outdoor activities", "rock climbing", "white-water rafting", "zip-lining", "surfing spots", "biking trails"])},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer=["historical landmarks", "ancient ruins", "UNESCO World Heritage Sites", "archaeological sites", "old towns", "monuments", "museums", "castles", "palaces", "heritage tours"])},          
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)
        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        expansion_list = json.loads(response)["answer"]
        
        if self.retriever_type == "sparse":
            # append original aspect description
            aspect_list = [query_aspect] * 5
            expansion_list = aspect_list + expansion_list
            expansions = ' '.join(expansion_list)
        
        else: # for dense retriever
            expansion_list = [query_aspect] + expansion_list
            expansions = '\n'.join(expansion_list)

        return expansions


    def _GQR(self, query_aspect: str) -> str:
        """
        GQR @ https://dl.acm.org/doi/pdf/10.1145/3589335.3651945
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate a list of three one sentence paraphrase of the aspect. 
        Provide your answers in valid JSON format with double quote: {{"answer": []}}.
        
        Aspect: {aspect}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer= ["ideal for those seeking adventurous experiences.", "perfect for adventure enthusiasts.", "a great choice for thrill-seekers."])},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer=["features notable historical landmarks.", "boasts a rich array of ancient sites.", "home to significant historical attractions."])},          
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]
        
        response = self.llm.generate(message)

        # parse the answer
        start = response.find("{")
        end = response.rfind("}") + 1
        response = response[start:end]
        paraphrase_list = json.loads(response)["answer"]
        
        if self.retriever_type == "sparse":
            # append original aspect description
            aspect_list = [query_aspect] * 3
            paraphrase_list = aspect_list + paraphrase_list
            paraphrases = ' '.join(paraphrase_list)
        
        else: # for dense retriever
            paraphrase_list = [query_aspect] + paraphrase_list
            paraphrases = '\n'.join(paraphrase_list)

        return paraphrases
    
    def _GenQREnsemble(self, query_aspect: str, n: int = 5):
        """
        GenQREnsemble @ https://arxiv.org/pdf/2404.03746
        """
        def rewrite_instruction(instruction: str, n: int):
            """
            """
            prompt = """
            Given the specified user instruction, paraphrase it to a list of {n} instructions. 
            Provide your answers in valid JSON format with double quote: {{"answer": []}}.

            Aspect: {instruction}
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

        instruction = "Optimize search results by suggesting meaningful expansion terms to enhance the following query aspect."
        instruction_list = rewrite_instruction(instruction=instruction, n=n)

        expansion_list = [query_aspect]
        for i in instruction_list:
            prompt = """
            {i}.
            Provide your answers in valid JSON format with double quote: {{"answer": []}}. 
            Aspect: {aspect}
            """
            message = [
                {"role": "system", "content": "You are a travel expert."},
                {"role": "user", "content": prompt.format(aspect=query_aspect, i=i)},
            ]

            response = self.llm.generate(message)

            start = response.find("{")
            end = response.rfind("}") + 1
            response = response[start:end]
            results = json.loads(response)["answer"]

            expansion_list += results
        
        if self.retriever_type == "sparse":
            expansion_list = [query_aspect] * 5 + expansion_list 
            ' '.join(expansion_list)
        else:
            expansion_list = list(set([e.lower() for e in expansion_list]))
            expansion_list = [query_aspect] + expansion_list
            expansions = '\n'.join(expansion_list)
        return expansions


def correct_and_extract(input_string) -> str:
    """
    Extracts the content within curly braces and corrects the 'answer' key-value pair.
    """
    pattern = r"\{([^}]*)\}"
    extracted_content = re.search(pattern, input_string)

    if not extracted_content:
        return "No content in curly braces found."

    content_within_braces = extracted_content.group(1)

    answer_pattern = r"(\s*\"?answer\"?\s*)(:)\s*(.*)"

    def replacer(match):
        key = '"answer":'  
        value = match.group(3).strip().strip("'\"")
        value = '"' + value.replace('"', '\\"') + '"'
        return f"{key} {value}"
    
    corrected_content = re.sub(answer_pattern, replacer, content_within_braces, flags=re.IGNORECASE)

    try:
        json_string = '{' + corrected_content + '}'
        corrected_dict = json.loads(json_string)
        return corrected_dict.get('answer', 'No valid "answer" found')
    except json.JSONDecodeError as e:
        return f'Correction failed; the string might still be invalid. Error: {str(e)}'
