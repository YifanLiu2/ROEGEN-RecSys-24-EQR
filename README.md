# Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation

This repository accompanies the paper titled "Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation," submitted to the Late-Breaking Result track at RecSys 2024.

## Dataset and Ground Truth
Our **TravelDest dataset** is meticulously curated to support this study, housed within the `data/` directory. Specific details include:
- `data/ground_truth`: This directory houses a JSON file containing the ground truth answers for each query listed in `data/general_queries.txt`.
- `data/wikivoyage_data_clean`: Contains cleaned metadata files sourced from Wikivoyage, useful for reference and further analysis.

## Setup
To get started, follow these steps:
1. **Configure API Key**: Copy the template configuration file and insert your OpenAI API key.
    ```bash
    cp config_template.py config.py
    # Open `config.py` and replace `"YOUR_API_KEY"` with your actual API key.
    ```
2. **Environment Setup**: Install necessary Python packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

## Execution
Run the project with ease using the script below. This script executes on `data/general_queries.txt` and outputs the results.
```bash
bash run.sh
# Results will be stored in the `output` folder.
