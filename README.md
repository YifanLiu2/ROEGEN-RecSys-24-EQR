# Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation

This repository provides the code and data accompanying our paper titled **"Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation"**.

## News
Our paper has been accepted to **The 1st Workshop on Risks, Opportunities, and Evaluation of Generative Models in Recommender Systems (ROEGEN@RecSys 2024)**. You can read the paper on [arXiv](https://arxiv.org/abs/2410.01598).

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

## Citation
@article{wen2024elaborative,
  title={Elaborative Subtopic Query Reformulation for Broad and Indirect Queries in Travel Destination Recommendation},
  author={Wen, Qianfeng and Liu, Yifan and Zhang, Joshua and Saad, George and Korikov, Anton and Sambale, Yury and Sanner, Scott},
  journal={arXiv preprint arXiv:2410.01598},
  year={2024}
}
