# Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation

This is the repository for the paper titled "Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation," submitted to the Late-Breaking Result track at RecSys 2024.

## Dataset and Ground Truth
Our **TravelDest dataset** is stored in the `data/` folder. The `data/ground_truth` directory contains the JSON file with the ground truth answers for each query listed in `data/general_queries.txt`. The `data/wikivoyage_data_clean` folder contains all metadata files from Wikivoyage.

## Setup
1. Configure your OpenAI API key:
    ```bash
    cp config_template.py config.py
    ```
    Open `config.py` and replace `"YOUR_API_KEY"` with your actual API key.

2. Set up the environment using the provided requirements file:
    ```bash
    pip install -r requirements.txt
    ```

## Execution
To run the default query, use the following command in your terminal:
    ```bash
    bash run.sh
    ```