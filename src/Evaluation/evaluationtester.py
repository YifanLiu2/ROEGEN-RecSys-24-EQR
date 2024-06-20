from RecallK import RecallK
# make some ground truth related to bruh.json
ground = {"Can you recommend cities with Disney attractions for my next vacation?":['Orlando', 'Burbank', 'Billund', 'Indre', 'Los Angeles', \
    'Puero Plata','Cancun', 'New Orleans', 'Myrtle Beach','Denver'], "which cities are known for being safe and welcoming for people traveling alone?": \
        ['Hargeisa', 'Mogadishu', 'Liberia', 'Chihuahua', 'Asturias'], "I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there?": \
        ['Bangkok', 'Beijing', 'Longyan'], "What cities in Europe host cultural festivals during the summer months that I shouldn't miss?": \
        ['Paris', 'Moncton', 'Boo', '888', 'AAAAAA'], "Seeking cities in tropical region suitable for a family vacation with kids.": \
        ['Billund', 'Puero Plaza', 'Montego Bay', 'Aruba', 'Bahamas', 'Key Largo']}
# recall should be 1, 1, 1, 0, 0.5

# sys = RecallK(k=10, json_path='output/bruh.json', ground_truths=ground)

# perquery, total = sys.recall_at_k()

# print(perquery)
# print(total)

# ground truth lengths are 10, 5, 3, 5, 6
# r precision should be 1, 1, 0.333, 0, 0.3333
from PrecisionR import PrecisionR
# sys2 = PrecisionR(json_path='output/dense_results_with_qe_v2.json', ground_truths=ground)
# perquery, total = sys2.precision_at_r()
# print(perquery)
# print(total)

# implement avg precision
# label : popular destination/big city/neither, can use gpt
# check ground truths: percentage in dataset
list1 = ['Orlando', 'Los Angeles', 'Paris', 'Tokyo', 'Shanghai']
list2 = ['Tokyo', 'Singapore', 'Osaka', 'Amsterdam', 'Toronto', 'Copenhagen', 'Seoul', 'Melbourne', 'Stockholm','San Francisco', 'Frankfurt', 'Los Angeles','Wellington','Zurich','Dallas','Taipei','Paris','Brussels', 'Barcelona', 'Dubai', 'Honolulu', 'Montreal', 'Reykjavik', 'Venice', 'Berlin', 'Doha']
list3 = ['Vientiane', 'Jaipur', 'Karachi','Srinagar','Kathmandu','Jakarta','Manila','Bangkok','Hanoi']
list4 = ['Pamplona','Edinburgh','Berlin','Rome','Budapest','London','Paris','Zagreb','Gothenburg']
list5 = ['Cancun','Mauritius','Myrtle Beach','Grenada','Providenciales','Roatan','Bonaire','Antigua','Bermuda','Aruba','Montego Bay']
newground = {"Can you recommend cities with Disney attractions for my next vacation?":list1, "which cities are known for being safe and welcoming for people traveling alone?": \
        list2, "I'm planning a trip to Asia on a budget. Any recommendations for budget-friendly cities there?": \
        list3, "What cities in Europe host cultural festivals during the summer months that I shouldn't miss?": \
        list4, "Seeking cities in tropical region suitable for a family vacation with kids.": \
        list5}

g1 = {"Seeking cities in tropical region suitable for a family vacation with kids.":list5, "Cities in tropical regions that are ideal for family vacations with kids often offer a blend of outdoor activities, child-friendly attractions, and relaxed environments. Singapore stands out as a prime destination with its clean, safe, and green spaces like Gardens by the Bay, along with family-centric attractions such as the Singapore Zoo and Universal Studios. San Juan, Puerto Rico, offers a mix of beach activities, historic sites like El Morro, and the interactive Museo del Ni帽o for younger travelers. Lastly, Cairns in Australia is another excellent choice, acting as a gateway to the Great Barrier Reef and providing educational yet fun experiences like the Kuranda Scenic Railway and Rainforestation Nature Park. These cities provide not only enjoyable but also enriching experiences for the whole family.":list5}

sys = RecallK(k=50, json_path='output/hybrid_results_with_qe_v2.json', ground_truths=newground)
perquery, total = sys.recall_at_k()
print(perquery)
print(total)

sys2 = PrecisionR(json_path='output/hybrid_results_with_qe_v2.json', ground_truths=newground)
perquery, total = sys2.precision_at_r()
print(perquery)
print(total)

# s1 = RecallK(k=50, json_path='output/dense_results_total_ela_top3_Tropical_family.json', ground_truths=g1)
# perquery, total = s1.recall_at_k()
# print(perquery)
# print(total)