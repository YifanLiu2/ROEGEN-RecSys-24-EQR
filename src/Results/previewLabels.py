import json
import os
import argparse
def preview_labels(labels_dir: str):
    # return a set of labels
    # if files are ending with labels.json
    labels_list = []
    for file in os.listdir(labels_dir):
        if file.endswith("labels.json"):
            # read the file
            with open(f"{labels_dir}/{file}", "r") as f:
                labels = json.load(f)
                labels_list.append(labels)
    return labels_list

def main():
    parser = argparse.ArgumentParser(description="Preview labels for each destination.")
    parser.add_argument("-l", "--labels_dir", type=str, required=True, help="Path to the directory containing labels files.")
    args = parser.parse_args()
    labels_list = preview_labels(args.labels_dir)
    # print the labels
    for labels in labels_list:
        for dest in labels:
            print(f"Destination: {dest}")
            print(f"Labels: {labels[dest]}")
            print("\n")