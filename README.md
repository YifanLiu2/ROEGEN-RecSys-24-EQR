# A Simple but Effective Elaborative Query Reformulation Approach for Natural Language Recommendation

This repository provides the code and data accompanying our paper titled **"A Simple but Effective Elaborative Query Reformulation Approach for Natural Language Recommendation"**.

## Dataset and Ground Truth
This repository includes three datasets for natural language recommendation evaluation, all located in the `data/` directory:

### TravelDest Dataset
- **Location**: `data/Traveldest/`
- **Contents**: 
  - `queries.txt`: 100 natural language queries for travel destinations
  - `ground_truth/`: Ground truth answers for each query
  - `corpus/`: Document corpus organized by city

### Yelp Restaurant Dataset
- **Location**: `data/Yelp_Restaurant/`
- **Contents**:
  - `queries.txt`: 100 restaurant-related queries
  - `ground_truth/`: Ground truth answers for evaluation
  - `corpus/`: Restaurant document corpus organized by city

### TripAdvisor Hotel Dataset
- **Location**: `data/TripAdvisor_Hotel/`
- **Contents**:
  - `queries.txt`: 100 hotel-related queries
  - `ground_truth/`: Ground truth answers for evaluation
  - `corpus/`: Hotel document corpus organized by city

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
Execute the main script using the command below. The script processes queries from the selected dataset and saves the results in the `output` directory:
```bash
bash run.sh
```

### Configuration
You can modify the parameters in `run.sh` to customize the execution:
- **`domain`**: Choose dataset - `"TripAdvisor_Hotel"`, `"Yelp_Restaurant"`, or `"Traveldest"`
- **`city`**: Target city for evaluation (e.g., `"chicago"`)
- **`mode`**: Query reformulation mode - `"none"`, `"q2e"`, `"gqr"`, `"q2d"`, `"eqr_5"`, `"eqr_8"`, `"eqr_10"`, `"eqr_12"`, `"eqr_15"`
- **`k`**: Number of documents to retrieve
- **`embedder_type`**: Embedding method - `"st"` (Sentence Transformers)
- **`retrieve_type`**: Retrieval method - `"dense"` or `"sparse"`

The script will automatically:
1. Process queries using the specified reformulation mode
2. Retrieve relevant documents using the configured retrieval method
3. Evaluate results using multiple metrics (R-Precision, Recall@K, MAP@K)
4. Save all results in the `output/{domain}/{mode}/{city}_{k}/` directory
