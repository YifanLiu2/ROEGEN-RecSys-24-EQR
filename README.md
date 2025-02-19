# Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation

This repository provides the code and data accompanying our paper titled **"Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation"**.

## Dataset and Ground Truth
We introduce the **TravelDest dataset**, curated specifically to support our study. It is located in the `data/` directory, which contains:
- **`data/ground_truth`**: A JSON file with the ground truth answers for each query listed in `data/general_queries.txt`.
- **`data/wikivoyage_data_clean`**: Cleaned metadata files sourced from Wikivoyage for reference and further analysis.

## Setup
To get started, follow these steps:
1. **Configure API Key**: Copy the template configuration file and insert your OpenAI API key:
    ```bash
    cp config_template.py config.py
    # Open `config.py` and replace "YOUR_API_KEY" with your actual API key.
    ```
2. **Environment Setup**: Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
Execute the main script using the command below. The script processes `data/general_queries.txt` and saves the results in the `output` directory:
```bash
bash run.sh
```
