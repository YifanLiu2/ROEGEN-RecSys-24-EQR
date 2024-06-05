import json, os
from tqdm import tqdm

from src.LLM.GPTChatCompletion import *
from src.Query.query import *

ANSWER_FORMAT = """{{"answer": {answer}}}"""

class queryProcessor:
    """
    Query Processor class
    """
    def __init__(self, query: QueryCE | list[QueryCE], llm: LLM, output_dir: str = "output"):
        """
        Initialize the query processor
        :param query:
        """
        if isinstance(query, QueryCE):
            self.query_list = [query]
        else:
            self.query_list = query

        self.llm = llm

        if not os.path.exists(output_dir):
            raise ValueError(f"Invalid output directory: {output_dir}")
        
        self.output_dir = output_dir
    

    def process_query(self) -> list[QueryCE]:
        """
        """
        results = []
        
        for query in tqdm(self.query_list, desc="Processing queries", unit="query"):
            query_result = {"query": query.description, "constraints": []}
            
            # extract constraints
            constraints = self._extract_constraints(query=query)
            for c_string in constraints:
                # classify
                is_specific = self._is_specific(c_string)
                is_verifiable = self._is_verifiable(c_string)

                # init constraint
                if is_specific:
                    c = SpecificConstraint(description=c_string, is_verifiable=is_verifiable)
                    query_result["constraints"].append({"description": c.description, "specificity": True, "verifiability": c.verifiable()})
                else:
                    c = GeneralConstraint(description=c_string, is_verifiable=is_verifiable)
                    # expand general constraint
                    self._expand_constraint(constraint=c)
                    query_result["constraints"].append({"description": c.description, "specificity": False, "verifiability": c.verifiable(), "expansions": c.expansion})
                
                # add constraints
                query.add_constraint(c)
            results.append(query_result)

        # store results
        json_string = json.dumps(results, indent=4)
        output_path = os.path.join(self.output_dir, "processed_query.json")
        with open(output_path, "w") as f:
            f.write(json_string)
        
        return self.query_list

    def _extract_constraints(self, query: QueryCE) -> list[str]:
        """
        """
        # actual prompt
        prompt = """
        Given the following query, generate a list of constraints specified in the query, in JSON format: {{"answer": []}}. 
        Each constraint should be in its minimal form and should not be further splittable.
        Each constraint should begin with 'The cities should'. 

        Query: {query}
        """

        # define answer format
        answer = ANSWER_FORMAT

        # 2-shots
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(query="Recommendation for cities with swimming spots near Orlando, Florida, for a refreshing day out?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should have swimming spots.", "The cities should be near Orlando, Florida."]))},
            {"role": "user", "content": prompt.format(query="Recommend me cities with historical sites and museums to explore during my travels?")},
            {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should have historical sites.", "The cities should have museums."]))},
            {"role": "user", "content": prompt.format(query=query.description)},
        ]

        response = self.llm.generate(message)

        try:
            start, end = response.find("{"), response.rfind("}") + 1
            constraint_list = json.loads(response[start:end])["answer"]
            return constraint_list

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print(f"Failed to extract constraints")
            print("GPT response: ", response)
            raise e


    def _is_specific(self, constraint: str) -> bool:
        """
        """
        prompt = """
        For a given constraint, determine whether the constraint contains well-defined terminology. Well-defined terminology refers to terms that are clear, unambiguous, and widely recognized, such as proper nouns like "Eiffel Tower" or "Amazon River".
        Follow the chain of thought by first identifying the specific term or feature the constraint applies to the cities. Then, determine whether the term is well-defined or not.
        Give your answer in JSON format: {{"answer": true | false}}.

        constraint: {constraint}
        """

        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(constraint="The cities should have the Disney Resort")},
            {"role": "assistant", "content": answer.format(answer="true")},
            {"role": "user", "content": prompt.format(constraint="The cities should have a museum")},
            {"role": "assistant", "content": answer.format(answer="false")},
            {"role": "user", "content": prompt.format(constraint="The cities should be budget-friendly")},
            {"role": "assistant", "content": answer.format(answer="true")},
            {"role": "user", "content": prompt.format(constraint=constraint)},
        ]

        response = self.llm.generate(message=message)

        try:
            start, end = response.find("{"), response.rfind("}") + 1
            is_specific = json.loads(response[start:end])["answer"]
            return is_specific

        except json.JSONDecodeError as e:
            print("Failed to parse JSON from response.")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print("Failed to classify the constraint")
            print("GPT response: ", response)
            raise e

    def _is_verifiable(self, constraint: str) -> bool:
        """
        """
        prompt = """
        For a given constraint, determine whether the constraint is a verifiable fact or a non-verifiable opinion. A verifiable fact is a statement that can be objectively confirmed through evidence or data, while a non-verifiable opinion is based on personal beliefs or feelings and cannot be proven true or false objectively.
        Give your answer in JSON format: {{"answer": true | false}}.

        Constraint: {constraint}
        """

        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(constraint="The cities should be near New York")},
            {"role": "assistant", "content": answer.format(answer="true")},
            {"role": "user", "content": prompt.format(constraint="The cities should have a museum")},
            {"role": "assistant", "content": answer.format(answer="true")},
            {"role": "user", "content": prompt.format(constraint="The cities should be budget-friendly")},
            {"role": "assistant", "content": answer.format(answer="false")},
            {"role": "user", "content": prompt.format(constraint=constraint)},
        ]

        response = self.llm.generate(message=message)
        try:
            start, end = response.find("{"), response.rfind("}") + 1
            is_verifiable = json.loads(response[start:end])["answer"]

            return is_verifiable

        except json.JSONDecodeError as e:
            print("Failed to parse JSON from response.")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print("Failed to classify the constraint")
            print("GPT response: ", response)
            raise e

    def _expand_constraint(self, constraint: GeneralConstraint) -> None:
        """
        """
        prompt = """
        For a given constraint, provide a list of 2 paraphrase of the term involved in the constraint that reflect a similar intent in JSON format: {{"answer": []}}.
        Each paraphrase should begin with "The cities should".

        Constraint: {constraint}
        """

        answer = ANSWER_FORMAT

        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.format(constraint="The cities should be budge friendly.")},
            {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should be affordable.", "The cities should offer cost-effective options."]))},
            {"role": "user", "content": prompt.format(constraint="The cities should have historical sites.")},
            {"role": "assistant", "content": answer.format(answer=json.dumps(["The cities should include historical landmarks.", "The cities should feature sites of historical significance."]))},
            {"role": "user", "content": prompt.format(constraint=constraint.description)},
        ]

        response = self.llm.generate(message)

        try:
            start, end = response.find("{"), response.rfind("}") + 1
            expansion_list = json.loads(response[start:end])["answer"]
            # add expanded constraint
            for expansion in expansion_list:
                constraint.expand(expansion)

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON from response")
            print("GPT response: ", response)
            raise e

        except Exception as e:
            print(f"Failed to expand constraints")
            print("GPT response: ", response)
            raise e