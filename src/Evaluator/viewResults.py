import pandas as pd
import glob
import os
import json
import argparse

def main(args):

    # Specify the directory containing the JSON files
    directory_path = args.directory_path

    # Use glob to list all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, '*.json'))

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Loop through each file
    for file_path in json_files:
        # Extract the metric from the filename
        metric = os.path.basename(file_path).replace('.json', '')
        
        # Load the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Convert the JSON data to a DataFrame
        temp_df = pd.DataFrame(list(data.items()), columns=['Query', metric])
        
        # If the main DataFrame is empty, initialize it with the current data
        if df.empty:
            df = temp_df
        else:
            # Otherwise, merge the new data with the existing DataFrame on the 'Query' column
            df = pd.merge(df, temp_df, on='Query', how='outer')

    # Save the final DataFrame to a CSV file
    output_csv_path = os.path.join(directory_path, 'evaluator_results_combined_metrics.csv')
    df.to_csv(output_csv_path, index=False)

    print(f"Combined metrics saved to {output_csv_path}")

    for file_path in json_files:
        # check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        os.remove(file_path)
        # print(f"Deleted {file_path}")
    print(f"Deleted all JSON files in {directory_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine evaluator results into a single CSV file')
    parser.add_argument('directory_path', type=str, help='Path to the directory containing the JSON files')
    args = parser.parse_args()
    main(args)
