
import json

# Step 1: Read JSON file and convert to Python object
with open('output/bruh.json', 'r') as json_file:
    data = json.load(json_file)

# Verify the loaded data
new_list = []

for e, n in data.items():
    new_list.append({})
    for _, m in n.items():
        new_list[-1][_] = m[0]

bigsort = []
for i in range(len(new_list)):
    sorted_dict_desc = dict(sorted(new_list[i].items(), key=lambda item: item[1], reverse=True)) 
    bigsort.append(sorted_dict_desc)

for i in range(len(bigsort)):
    print(list(data.keys())[i]) 
    for j in range(10):
        print(list(bigsort[i].items())[j])