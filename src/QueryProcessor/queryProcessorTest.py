import pickle
from config import API_KEY
from .queryProcessor import *

# init query
query_string_list = ["Can you recommend cities with Disney attractions for my next vacation?","which cities are known for being safe and welcoming for people traveling alone?","I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there?","What cities in Europe host cultural festivals during the summer months that I shouldn't miss?","Seeking cities in tropical region suitable for a family vacation with kids."]
# query_string_list = ["I'm a huge Harry Potter fan. Where are some cities known for Harry Potter film?", "As a solo traveler, which cities are known for being safe and welcoming for people traveling alone?", "I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there?", "Seeking cities in tropical region suitable for a family vacation with kids. Any suggestions?", "What cities host cultural festivals during the summer months that I shouldn't miss?"]
queries = [QueryCE(description=query_string) for query_string in query_string_list]

# process query
gpt = GPTChatCompletion(api_key=API_KEY)
query_processor = queryProcessor(query=queries, llm=gpt)
q_list = query_processor.process_query()

with open("output/processed_query.pkl", "wb") as file:
    pickle.dump(q_list, file)