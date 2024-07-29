import pandas as pd
import os, json, argparse, glob, shutil




def make_csv(output_dir):
    # merge all json file
    json_files = glob.glob(os.path.join(output_dir, '*.json'))
    df = pd.DataFrame()
    for file_path in json_files:
        metric = os.path.basename(file_path).replace('.json', '')
        
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        temp_df = pd.DataFrame(list(data.items()), columns=['Query', metric])
        
        if df.empty:
            df = temp_df
        else:
            df = pd.merge(df, temp_df, on='Query', how='outer')

    # save result
    save_dir = os.path.dirname(output_dir)
    output_csv_path = os.path.join(save_dir, 'evaluator_results_combined_metrics.csv')
    df.to_csv(output_csv_path, index=False)

    # remove all json file
    shutil.rmtree(output_dir)


def main(args):
    output_dir = args.output_dir
    make_csv(output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine evaluator results into a single CSV file')
    parser.add_argument('--output_dir', type=str, help='Path to the directory containing the JSON files')
    args = parser.parse_args()
    main(args)
