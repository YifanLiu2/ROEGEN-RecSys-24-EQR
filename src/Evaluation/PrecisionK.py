from BaseEvaluator import Evaluator
from fuzzywuzzy import fuzz
import json

class PrecisionK(Evaluator):

    def __init__(self, k: int, json_path, ground_truths:dict): # format of the ground truth is {question: [city, city, city], question: [city, city, city]}
        super().__init__(json_path, ground_truths, k)
        self.data = None
        with open(self.json_path, 'r') as json_file:
            self.data = json.load(json_file)

        self.new_list = []

        for question, questionitems in self.data.items():
            self.new_list.append({})
            for city, cityitems in questionitems.items():
                self.new_list[-1][city] = cityitems[0] # cityitems[0] = score
        # new_list = [{city:score, city:score...}, {}, {}]

        for i in range(len(self.new_list)):
            self.new_list[i] = dict(sorted(self.new_list[i].items(), key=lambda item: item[1], reverse=True))

        # new_list = [{}, {}, {}, {}, {}] for 5 prompts
        self.master = {} # format will be same as ground truth: {question: {city:score, city:score...}, question: {city, score...}, ..}
        for i in range(len(self.new_list)):
            self.master[list(self.data.keys())[i]] = self.new_list[i]

# for question, citydict in master.items():
#     print(question) 
#     for j in range(10):
#         print(list(citydict.items())[j][0]) # gives the city

    def precision_at_k(self):
        # get per query and overall and return both
        recall_per_query = {}
        recall_overall = 0

        for question, citydict in self.master.items():

            truthlist = [element.lower() for element in self.ground_truths[question]]
            top10list = [list(citydict.items())[j][0].lower() for j in range(self.k)] # [city1, city2... , city10]

            correct = 0
            for d in top10list:
                for d2 in truthlist:
                    if fuzz.ratio(d, d2) > 85:
                        if d != d2:
                            print(f"Matched {d} to {d2} with {fuzz.ratio(d, d2)}% similarity.")
                        correct += 1
                        break
            
            recall_per_query[question] = correct / self.k
            recall_overall += correct / self.k
        
        recall_overall /= len(self.master)
        return recall_per_query, recall_overall