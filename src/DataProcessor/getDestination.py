import ast
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
from time import sleep

def load_city_list(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        city_list = ast.literal_eval(content)
        return list(city_list)

def extract_city_names(city_list):
    return [city.split(',')[0].strip() for city in city_list]

def find_closest_cities(list1, list2):
    city_names_list1 = extract_city_names(list1)
    closest_cities = []
    differences = []
    index = 0
    for city in tqdm(city_names_list1, desc="Finding closest cities"):

        closest_match, score = process.extractOne(city.lower(), list2, scorer=fuzz.token_sort_ratio)
        if score != 100:
            differences.append(tuple([list1[index], closest_match]))
        else:
            closest_cities.append(closest_match)
        index += 1
    return closest_cities, differences

def main():
    # Load list1 and extract city names
    list1 = load_city_list('data/list1.txt')
    # Load list2
    list2 = load_city_list('data/list2.txt')
    
    # Find closest cities for sorted list1
    closest_cities, differences = find_closest_cities(list1, list2)
    
    with open('closest_cities.txt', 'w') as file:
        for city in closest_cities:
            file.write(f"{city}\n")

    # Save differences
    with open('differences.txt', 'w') as file:
        for city in differences:
            file.write(f"{city}\n")

if __name__ == "__main__":
    main()
