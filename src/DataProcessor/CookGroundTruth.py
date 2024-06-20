import os
import json
import argparse
def main(args):
    output_dir = args.output_dir
    queries_name = ["Seeking cities for a family vacation.", 
                    "Which cities offer the best culinary experiences for food lovers?",
                    "What cities are best for a romantic getaway?",
                    "What are the top travel cities for adventure seekers?",
                    "Which cities are ideal for art and culture enthusiasts?",
                    "Can you suggest cities that are great for winter sports?",
                    "I want to find cities with good theme parks."]
    ground_truth_dir = args.ground_truth_dir
    ground_truth = assemble_ground_truth(ground_truth_dir, queries_name)
    with open(f"{output_dir}/ground_truth.json", "w") as file:
        json.dump(ground_truth, file, indent=4)


def assemble_ground_truth(ground_truth_dir: dir, queries_name: list[str]):
    # read all json files under the directory
    new_ground_truth = {}
    index = 0
    for file in os.listdir(ground_truth_dir):
        if file.endswith(".json"):
            file = os.path.join(ground_truth_dir, file)
            with open(file, 'r') as json_file:
                data = json.load(json_file) # the format of the ground truth is {city: 1 or 0}
                # get all valid cities
                valid_cities = [city for city, value in data.items() if value == 1]
                # query name is the file name
                query_name = queries_name[index].lower()

                new_ground_truth[query_name] = valid_cities
                index += 1
    
    return new_ground_truth

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--ground_truth_dir", required=True, help="Path to the ground truth directory")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to the output file")
    args = parser.parse_args()
    main(args=args)