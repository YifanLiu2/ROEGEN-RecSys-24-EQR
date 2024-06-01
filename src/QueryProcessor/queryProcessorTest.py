from queryProcessor import queryProcessor
import json
import os
os.environ["OPENAI_API"] = "Your API Key Here"
queries = ["I want to go to disney", "Find me some places with good beach"]
QueryProcessor = queryProcessor(queries)
QueryProcessor.aspectExtraction()
with open("extracted_query_aspects.json", "r") as f:
    print(json.load(f))
