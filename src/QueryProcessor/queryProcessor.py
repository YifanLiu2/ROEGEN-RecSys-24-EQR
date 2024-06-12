import json, os, re, pickle
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
    def __init__(self, input_path: str, llm: LLM, mode_name: str = None, output_dir: str = "output"):
        """
        Initialize the query processor
        :param query:
        :param llm:
        :param mode_name: can only be "expand", "reformulate", "elaborate"
        :param output_dir:
        """ 
        self.llm = llm

        if mode_name is not None and mode_name not in MODE:
            raise ValueError(f"Invalid mode name: {mode_name}, could only be {MODE}")

        if not input_path.endswith('.txt'):
            raise ValueError(f"Invalid file type: {input_path} is not a .txt file")
        
        os.makedirs(output_dir, exist_ok=True)

        self.mode_name = mode_name
        self.output_dir = output_dir
        self.query_list = self._load_queries(input_path)
    
    def process_query(self) -> list[Query]:
        """
        Process the queries
        """
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
            
            # fetch aspect processing function
            aspect_processing_func = None
            if self.mode_name == "reformulate":
                aspect_processing_func = self._reformulate_aspect
            elif self.mode_name == "expand":
                aspect_processing_func = self._expand_aspect
            elif self.mode_name == "elaborate":
                aspect_processing_func = self._elaborate_aspect
            
            if aspect_processing_func is not None:
                for aspect in curr_query.get_all_aspects():
                    new_desc = aspect_processing_func(query_aspect=aspect.description)
                    aspect.set_new_description(new_description=new_desc)
            
            result_queries.append(curr_query)
        
        self._save_results(result=result_queries)
        return result_queries
    
    def _load_queries(self) -> list[str]:
        """
        """
        with open(self.input_path, "r") as f:
            queries = [line.strip().lower() for line in f]
        return queries
    
    def _save_results(self, result: list[Query]):
        """
        """
        file_path = os.path.join(self.output_dir, f"processed_query_{self.mode_name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(result, f)
        
    def _extract_aspects(self, query: str) -> tuple[list[str], list[str], list[str]]:
        """
        Extract preferences and constraints from the query
        """
        # actual prompt
        # Corrected prompt string
        prompt = """
        Given a user's query about travel destination city recommendation, please categorize each aspect in the query to the following types:

        1. “constraints” - This category requires using a Checker to confirm the presence of highly specific and rare attributes in a city, suitable for binary (yes/no) decisions. For instance, attributes like being located in a specific continent (e.g., Europe) or having a Disney attraction are considered valid constraints because only a limited number of cities possess these qualities. Conversely, attributes such as the presence of museums, historical sites, or cultural festivals in June are not considered constraints due to their commonality or ambiguity in numerous cities. Only attributes that clearly distinguish a small subset of cities are categorized under constraints.

        2. “preferences” - This category requires a Reasoner tool to assess city attributes that are inherently subjective and cannot be reduced to binary, objective statements. For example, evaluating whether a city is "suitable for romantic activities" fits this category because it depends heavily on personal interpretation and cannot be objectively measured. In contrast, the attribute "known for historical sites" does not qualify, even though it might appear subjective. This is because it can be simplistically converted into a binary form such as "has historical sites," which involves a straightforward verification, thus excluding it from this category. 

        3. "hybrids” - This category requires both a Checker and a Reasoner and addresses city attributes that can readily convert between objective and subjective forms. For instance, "has museums" is an objective fact that can easily transition into a subjective assessment as "known for museums," reflecting the city’s cultural stature based on its museums. The conversion also works in reverse; the subjective reputation of being "known for museums" can be substantiated by objectively verifying that the city indeed has museums. However, an attribute like "safety" does not fit this category. It cannot be effectively converted into objective metrics.

        Requirements:
        - Each aspect should be expressed in its minimal form and should not be further divisible to ensure clarity and precision in the recommendation process.
        - provide your answer strictly in JSON format: {{\"answer\": {{\”constraints\”: [], \”preferences\”: [], \"hybrids\”: []}}}}
        
        Query: {query}
        """
        # define answer format
        answer = ANSWER_FORMAT

        # few-shots
        message = [
            {"role": "system", "content": "You are a travel expert."},
            {"role": "user", "content": prompt.format(query="Recommend me cities with historical sites and museums to explore during my travels?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraints": [],"preferences": [],"hybrids": ["historical sites", "museums"]}}))},
            {"role": "user", "content": prompt.format(query="Looking for cities with for a romantic honeymoon. Any suggestions?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraints": [],"preferences": ["suitable for romantic honeymoon"],"hybrids": []}}))},
            {"role": "user", "content": prompt.format(query="I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there? ")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraints": ["in Asia"], "preferences": ["on a budget"], "hybrids": []}}))},
            {"role": "user", "content": prompt.format(query="I want a city with a major film festival in June and good seafood restaurants.")},
            {"role": "assistant", "content": answer.format(answer=json.dumps({"answer": {"constraints": [], "preferences": ["good seafood restaurants"], "hybrids": ["major film festival in June"]}}))},
            {"role": "user", "content": prompt.format(query=query)},
        ]

        response = self.llm.generate(message)
        
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            answer = json.loads(response[start:end])["answer"]
            preferences = answer["preferences"]
            constraints = answer["constraints"]
            hybrids = answer["hybrids"]
            return preferences, constraints, hybrids

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
        return correct_and_extract(response)
        

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

         # parse response
        return correct_and_extract(response)

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
        return correct_and_extract(response)
        

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
