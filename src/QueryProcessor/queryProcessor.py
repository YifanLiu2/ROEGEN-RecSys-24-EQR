import json, os
from tqdm import tqdm

from src.LLM.GPTChatCompletion import *
from src.Entity.query import *
from src.Entity.aspect import *

MODE = {"expand", "reformulate", "elaborate"}
ANSWER_FORMAT = """{{"answer": {answer}}}"""

class queryProcessor:
    """
    Query Processor class
    """
    def __init__(self, query: str | list[str], llm: LLM, mode_name: str = None, output_dir: str = "output"):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param mode_name: can only be "expand", "reformulate", "elaborate"
        :param output_dir:
        """
        if isinstance(query, str):
            self.query_list = [query]
        else:
            self.query_list = query
            
        self.mode_name = mode_name
        self.llm = llm

        if mode_name is not None and mode_name not in MODE:
            raise ValueError(f"Invalid mode name: {mode_name}, could only be {MODE}")

        if not os.path.exists(output_dir):
            raise ValueError(f"Invalid output directory: {output_dir}")
        
        self.output_dir = output_dir
    

    def process_query(self) -> list[Query]:
        """
        Process the queries
        """
        result_queries = []

        for query in tqdm(self.query_list, desc="Processing queries", unit="query"):
            query_result = {"query": query, "preferences": [], "constraints": []}
            curr_query = Query(description=query)
            
            # fetch aspect processing function
            aspect_processing_func = None
            if self.mode_name == "reformulate":
                aspect_processing_func = self._reformulate_aspect
            elif self.mode_name == "expand":
                aspect_processing_func = self._expand_aspect
            elif self.mode_name == "elaborate":
                aspect_processing_func = self._elaborate_aspect
            
            if aspect_processing_func is not None:
                # extract preferences and constraints
                preferences, constraints = self._extract_aspects(query=query)
                query_result["preferences"] = preferences
                query_result["constraints"] = constraints

                # preferences
                for p in preferences:
                    reformulation = aspect_processing_func(query_aspect=p)
                    preference = Preference(description=p)
                    preference.set_new_description(reformulation)
                    curr_query.add_preference(preference)

                # constraints
                for c in constraints:
                    reformulation = aspect_processing_func(query_aspect=c)
                    constraint = Constraint(description=c)
                    constraint.set_new_description(reformulation)
                    curr_query.add_constraint(constraint)
            
            result_queries.append(curr_query)
            
        return result_queries
       

    def _extract_aspects(self, query: str) -> tuple[list[str], list[str]]:
        """
        Extract preferences and constraints from the query
        """
        # actual prompt
        # Corrected prompt string
        prompt = """
        Given the following query for travel cities recommendations, generate a list of constraints and preferences in JSON format: {{\"answer\": {{\"preferences\": [], \"constraints\": []}}}}.
        A 'constraint' is a requirement that must be met and typically describes a verifiable truth about the cities. 
        A 'preference' is a desirable, subjective feature for the cities that is not necessarily verifiable.    

        Each constraint or preference should be in its minimal form and should not be further splittable.

        Query: {query}
        """
        # define answer format
        answer = ANSWER_FORMAT

        # few-shots
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Recommend me cities with historical sites and museums to explore during my travels?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"preferences": [], "constraints": ["has historical sites", "has museums"]}}))},
            {"role": "user", "content": prompt.format(query="Looking for cities with for a romantic honeymoon. Any suggestions?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"preferences": ["suitable for romantic honeymoon"], "constraints": []}}))},
            {"role": "user", "content": prompt.format(query="I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there? ")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"preferences": ["budget-friendly"], "constraints": ["in Asia"]}}))},
            {"role": "user", "content": prompt.format(query=query)},
        ]

        response = self.llm.generate(message)
        
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            answer = json.loads(response[start:end])["answer"]
            preferences = answer["preferences"]
            constraints = answer["constraints"]
            return preferences, constraints

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print(f"Failed to extract constraints")
            print("GPT response: ", response)
            raise e


    def _reformulate_aspect(self, query_aspect: str) -> str:
        """
        Reformulate one aspect of the query
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate one sentence reformulation of the aspect to better reflect the user's intent. 
        Provide your answers in valid JSON format with double quote: {{"answer": "YOUR ANSWER"}}.
        
        Aspect: {aspect}
        """

        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer="Which cities are best suited for adventure seekers looking for thrilling activities?")},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer="Which cities are rich in historical sites and cultural heritage?")},          
            {"role": "user", "content": prompt.format(aspect="on Asia")},
            {"role": "assistant", "content": answer.format(answer="Which cities in Asia are recommended for travelers?")},  
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]
        
        response = self.llm.generate(message)

        # parse response
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            reformulation = json.loads(response[start:end])["answer"]
            return reformulation
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e
        

    def _expand_aspect(self, query_aspect: str) -> str:
        """
        Expand one aspect of the query
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate a list of three similar but improved descriptions of the aspect to better reflect the user's intent. 
        Provide your answers in valid JSON format with double quote: {{"answer": []}}.

        Aspect: {aspect}
        """
        # define answer format
        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(aspect="suitable for adventure seekers")},
            {"role": "assistant", "content": answer.format(answer=["Ideal for outdoor adventure enthusiasts.", "Great for those seeking thrilling adventure activities.", "Perfect for adventure seekers looking for excitement and challenges."])},
            {"role": "user", "content": prompt.format(aspect="has historical sites")},
            {"role": "assistant", "content": answer.format(answer=["Rich in historical landmarks and museums.", "Offers a wealth of cultural and historical sites.", "Filled with historical attractions and heritage sites."])},          
            {"role": "user", "content": prompt.format(aspect="on Asia")},
            {"role": "assistant", "content": answer.format(answer=["Located in Asia.", "Situated in Asia.", "In Asia."])},  
            {"role": "user", "content": prompt.format(aspect=query_aspect)},
        ]

        response = self.llm.generate(message)

        try:
            start, end = response.find("{"), response.rfind("}") + 1
            expansion_list = json.loads(response[start:end])["answer"]
            expansion_list.append(query_aspect)
            joined_expansion = " ".join(expansion_list)
            return joined_expansion
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e
            

    def _elaborate_aspect(self, query_aspect: str) -> str:
        """
        Elaborate one aspect of the query
        """
        prompt = """
        Given the specified aspect of a user's travel cities recommendation query, generate one paragraph specific definition of the aspect. 
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
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            definition = json.loads(response[start:end])["answer"]
            return definition
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e
        
    # def _define_constraints(self, constraint: Constraint) -> Constraint:
    #     """
    #     """
    #     prompt = """
    #     You are a travel expert, please give a specific definition on this constraint, in JSON format: {{"answer": []}}.
    #     Your answer should be in a short paragraph.

    #     Constraint: {constraint}
    #     """

    #     # define answer format
    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt.format(constraint="a place be affordable.")},
    #         {"role": "assistant", "content": answer.format(answer=json.dumps("An affordable place is typically defined as a location where the cost of living or the price of specific services and commodities (like housing, food, and transportation) is relatively low compared to the average income or budget constraints of an individual or family. In more concrete terms, a place might be considered affordable if housing costs do not exceed 30 percents of a household's income, which is a common benchmark used by economists and urban planners to gauge housing affordability. This concept can extend to other expenses, suggesting that an affordable place has a cost of living index lower than the national average, making it financially manageable for residents with average or below-average incomes."))},
    #         {"role": "user", "content": prompt.format(constraint="a place with Disney")},
    #         {"role": "assistant", "content": answer.format(constraint=json.dumps("When considering cities for a vacation that feature Disney attractions, it's beneficial to explore a variety of options worldwide that offer unique Disney experiences. This exploration could include well-known destinations like Orlando and Anaheim, which are famous for their expansive Disney theme parks, such as Waltl Disney World and Disneyland. Additionally, international locations such as Paris, Tokyo, Hong Kong, and Shanghai also host Disney resorts, each providing distinctive attractions and cultural twists on the classic Disney formula. Understanding the specific attractions, seasonal events, and accommodation options available at each location can significantly influence the decision-making process, ensuring a magical and well-suited vacation for families, Disney enthusiasts, or anyone looking to immerse themselves in the enchanting world of Disney."))},
    #         {"role": "user", "content": prompt.format(answer=constraint.description)},
    #     ]

    #     response = self.llm.generate(message)

    #     try:
    #         start, end = response.find("{"), response.rfind("}") + 1
    #         definition_str = json.loads(response[start:end])["answer"]
    #         constraint.define(definition_str)
    #         return constraint

    #     except json.JSONDecodeError as e:
    #         print(f"Failed to parse JSON from response")
    #         print("GPT response: ", response)
    #         raise e

    #     except Exception as e:
    #         print(f"Failed to extract constraints")
    #         print("GPT response: ", response)
    #         raise e


    # def _is_specific(self, constraint: str) -> bool:
    #     """
    #     """
    #     prompt = """
    #     For a given constraint, determine whether the constraint contains well-defined terminology. Well-defined terminology refers to terms that are clear, unambiguous, and widely recognized, such as proper nouns like "Eiffel Tower" or "Amazon River".
    #     Follow the chain of thought by first identifying the specific term or feature the constraint applies to the cities. Then, determine whether the term is well-defined or not.
    #     Give your answer in JSON format: {{"answer": true | false}}.

    #     constraint: {constraint}
    #     """

    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt.format(constraint="a place have the Disney Resort")},
    #         {"role": "assistant", "content": answer.format(answer="true")},
    #         {"role": "user", "content": prompt.format(constraint="a place have a museum")},
    #         {"role": "assistant", "content": answer.format(answer="false")},
    #         {"role": "user", "content": prompt.format(constraint="a place be budget-friendly")},
    #         {"role": "assistant", "content": answer.format(answer="true")},
    #         {"role": "user", "content": prompt.format(constraint=constraint)},
    #     ]

    #     response = self.llm.generate(message=message)

    #     try:
    #         start, end = response.find("{"), response.rfind("}") + 1
    #         is_specific = json.loads(response[start:end])["answer"]
    #         return is_specific

    #     except json.JSONDecodeError as e:
    #         print("Failed to parse JSON from response.")
    #         print("GPT response: ", response)
    #         raise e

    #     except Exception as e:
    #         print("Failed to classify the constraint")
    #         print("GPT response: ", response)
    #         raise e

    # def _is_verifiable(self, constraint: str) -> bool:
    #     """
    #     """
    #     prompt = """
    #     For a given constraint, determine whether the constraint is a verifiable fact or a non-verifiable opinion. A verifiable fact is a statement that can be objectively confirmed through evidence or data, while a non-verifiable opinion is based on personal beliefs or feelings and cannot be proven true or false objectively.
    #     Give your answer in JSON format: {{"answer": true | false}}.

    #     Constraint: {constraint}
    #     """

    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt.format(constraint="a place be near New York")},
    #         {"role": "assistant", "content": answer.format(answer="true")},
    #         {"role": "user", "content": prompt.format(constraint="a place have a museum")},
    #         {"role": "assistant", "content": answer.format(answer="true")},
    #         {"role": "user", "content": prompt.format(constraint="a place be budget-friendly")},
    #         {"role": "assistant", "content": answer.format(answer="false")},
    #         {"role": "user", "content": prompt.format(constraint=constraint)},
    #     ]

    #     response = self.llm.generate(message=message)
    #     try:
    #         start, end = response.find("{"), response.rfind("}") + 1
    #         is_verifiable = json.loads(response[start:end])["answer"]

    #         return is_verifiable

    #     except json.JSONDecodeError as e:
    #         print("Failed to parse JSON from response.")
    #         print("GPT response: ", response)
    #         raise e

    #     except Exception as e:
    #         print("Failed to classify the constraint")
    #         print("GPT response: ", response)
    #         raise e

    # def _expand_constraint(self, constraint: GeneralConstraint) -> None:
    #     """
    #     """
    #     prompt = """
    #     For a given constraint, provide a list of 2 paraphrase of the term involved in the constraint that reflect a similar intent in JSON format: {{"answer": []}}.
    #     Each paraphrase should begin with "The cities should".

    #     Constraint: {constraint}
    #     """

    #     answer = ANSWER_FORMAT

    #     message = [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt.format(constraint="The cities should be budge friendly.")},
    #         {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should be affordable.", "The cities should offer cost-effective options."]))},
    #         {"role": "user", "content": prompt.format(constraint="The cities should have historical sites.")},
    #         {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should include historical landmarks.", "The cities should feature sites of historical significance."]))},
    #         {"role": "user", "content": prompt.format(constraint=constraint.description)},
    #     ]

    #     response = self.llm.generate(message)

    #     try:
    #         start, end = response.find("{"), response.rfind("}") + 1
    #         expansion_list = json.loads(response[start:end])["answer"]
    #         # add expanded constraint
    #         for expansion in expansion_list:
    #             constraint.expand(expansion)

    #     except json.JSONDecodeError as e:
    #         print(f"Failed to parse JSON from response")
    #         print("GPT response: ", response)
    #         raise e

    #     except Exception as e:
    #         print(f"Failed to expand constraints")
    #         print("GPT response: ", response)
    #         raise e