
import json

# Step 1: Read JSON file and convert to Python object
with open('output/dense_results_total_ela_top3_Tropical_family.json', 'r') as json_file:
    data = json.load(json_file)

# Verify the loaded data
new_list = []

for e, n in data.items():
    new_list.append({})
    for _, m in n.items():
        new_list[-1][_] = m[0]


for i in range(len(new_list)):
    new_list[i] = dict(sorted(new_list[i].items(), key=lambda item: item[1], reverse=True)) 
    # bigsort.append(sorted_dict_desc)

master = {}

for i in range(len(new_list)):
    master[list(data.keys())[i]] = new_list[i]

for question, citydict in master.items():
    print(question) 
    for j in range(50):
        # print(list(citydict.items())[j])
        print(list(citydict.items())[j][0])

# for i in range(len(new_list)):
#     print(list(data.keys())[i]) 
#     for j in range(10):
#         print(list(new_list[i].items())[j])