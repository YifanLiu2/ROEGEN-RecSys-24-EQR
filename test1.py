import os
def get_txt_filenames(directory):
    filenames = set()
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            filenames.add(os.path.splitext(file)[0])
    return filenames
import json
with open('output/theme_parks.json', 'r', encoding='utf-8') as json_file:
    d = json.load(json_file)


dest_names = get_txt_filenames("data/clean_destination_air_canada_xml")
yes = 0
no = 0
for key in d:
    if key in dest_names: 
        yes += 1
    else: 
        print(key)
        no += 1
print(yes, no)