import json
import os
from src.LLM.GPTChatCompletion import GPTChatCompletion


class queryProcessor:
    """
    Query Processor class
    """
    def __init__(self, query: str | list[str]):
        """
        Initialize the query processor
        :param query:
        """
        if isinstance(query, str):
            self.query = [query]
        else:
            self.query = query
        self.llm = GPTChatCompletion(api_key=os.getenv("OPENAI_API"))
        self.queriesAspects = dict()

    def aspectExtraction(self) -> None:
        """
        Use LLM to extract aspects from the query
        :return:
        """
        aspect_extraction_prompt = '''
        Given the following query, generate aspect phrases that describe the intent of the query. Include important aspects from the query and additional related terms.
        Return the list of aspects in the JSON format like this:
        {{
            "aspects": ["aspect1", "aspect2", "aspect3"]
        }}

        Query: {query}
        '''

        for q in self.query:
            extracted_aspects = self.llm.generate(aspect_extraction_prompt.format(query=q), max_tokens=4000)
            try:
                start, end = extracted_aspects.find("{"), extracted_aspects.rfind("}") + 1
                extracted_aspects = json.loads(extracted_aspects[start:end])
                # Store the extracted aspects in a dictionary, where the key is the query and the value is the label
                aespectCategory = {}
                for aspect in extracted_aspects["aspects"]:
                    # Classify the aspect as a soft or hard constraint using classifyAspects method
                    aespectCategory[aspect] = self.classifyAspects(aspect, q)
                self.queriesAspects[q] = aespectCategory
            except Exception as e:
                print(f"Failed to extract aspects for {self.query}: {e}")
                print("GPT output: ", extracted_aspects)

        # Store the extracted aspects in a JSON file in the data folder outside the src folder
        with open("extracted_query_aspects.json", "w") as f:
            json.dump(self.queriesAspects, f)

    def classifyAspects(self, aspect: str, originalQuery: str) -> str:
        """
        Classify the extracted aspects whether they are soft or hard constraints
        :return:
        """
        classify_aspect_prompt = '''
        For a travel recommendation system, do you think the following aspect is a soft(paraphrasable) or hard(not paraphrasable) constraint for the query?
        Return the classification directly in the JSON format like this:
        {{
            "classification": "soft / hard"
        }}
        Query: {query}
        Aspect: {aspect}
        '''
        classification = self.llm.generate(classify_aspect_prompt.format(query=originalQuery, aspect=aspect), max_tokens=4000)
        # Get the string out of the JSON response
        try:
            start, end = classification.find("{"), classification.rfind("}") + 1
            classification = json.loads(classification[start:end])["classification"]
        except Exception as e:
            print(f"Failed to classify aspect {aspect} for query {originalQuery}: {e}")
            print("GPT output: ", classification)
        return classification
