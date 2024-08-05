import pandas as pd
import os, json, argparse


import pandas as pd
import os

def combine_csv(output_dir):
    output_dir = args.output_dir
    entries = os.listdir(output_dir)

    # get all csv file in directory
    directories = [entry for entry in entries if os.path.isdir(os.path.join(output_dir, entry))]
    csv_files = []
    for directory in directories:
        directory_path = os.path.join(output_dir, directory)
        for file in os.listdir(directory_path):
            if file.endswith('.csv'):
                full_file_path = os.path.join(directory_path, file)
                csv_files.append(full_file_path)

    if len(csv_files) != len(directories):
        raise ValueError("Invalid number of .csv files compared to directories.")

    # concatenate the csv file
    dataframes = []
    for csv_file, directory in zip(csv_files, directories):
        df = pd.read_csv(csv_file)

        # rename the columns
        rename_columns = {col: f"{col}_{directory}" for col in df.columns[1:]}  
        df.rename(columns=rename_columns, inplace=True)
        
        df.set_index(df.columns[0], inplace=True)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, axis=1, join='outer')

    # calculate statistics
    mean_row = merged_df.mean()
    mean_df = pd.DataFrame([mean_row], index=['mean'])
    final_df = pd.concat([mean_df, merged_df])

    # save
    final_output_path = os.path.join(output_dir, 'evaluation_results.csv')
    final_df.to_csv(final_output_path)


def main(args):
    output_dir = args.output_dir
    combine_csv(output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine evaluator results into a single CSV file')
    parser.add_argument('--output_dir', type=str, help='Path to the directory containing the JSON files')
    args = parser.parse_args()
    main(args)