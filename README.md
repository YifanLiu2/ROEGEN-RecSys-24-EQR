# Travel Destination Recommender 
This is an LLM-based recommender for travel destinations. 

## Set-up
Run the following command line to set up enviroment:
```bash
conda create -n PyTravel python=3.9
```

## Embedder 
The embedder encodes destination text data scraped from Wiki-travel into high-dimensional vector representations, using either a GPT-based model or a Sentence-Transformer based model.

### Usage 
Run the following command line:
```bash
python -m src.Embedder.embedderRunner -d data/clean_destination_air_canada_xml -o output --split_type <sentence_or_section> --emb_type <gpt_or_st>
```
- `-d, --data_path`: Path to the directory containing text files.
- `-o, --output_type`: Path where embeddings should be saved.
- `--split_type`: The type of text splitting to apply before embedding. Choices are 'sentence' or 'section'.
- `--emb_type`: Specify the type of the embedder. Available types are 'gpt' or 'st'.

## Query Processor
The query processor analyzes a given user query by breaking it down into a list of aspects and then applies different processing strategies to each aspect to better capture and reflect the user's intent:

- **Reformulation**: Rewrites the aspect description.
- **Expansion**: Generates three improved descriptions and concatenates these with the original aspect.
- **Elaboration**: Generate a detailed paragraph that elaborates on the original aspect.

### Usage 
Run the following command line:
```bash
python -m src.QueryProcessor.queryProcessorRunner -i data/queries.txt -o output --mode <reformulate_or_expand_or_elaborate>
```

- `-i, --input_path`: Specifies the path to the input file containing the user queries. This file should be in text format, with each query on a new line.

- `-o, --output_dir`: Defines the directory where the processed queries will be saved.

- `--mode`: Determines the processing strategy to apply to each query. There are three available options: 'reformulate', 'expand', or 'elaborate'.


## Retriever
The retriever extracts relevant text chunks from destination texts based on processed user queries.

### Usage 
Run the following command line:
```bash
python -m src.Retriever.retrieverRunner -q query/path -e embeddings/path -o output/path --emb_type <gpt_or_st>
```

- `-q, --query_path`: Path to the input file containing processed queries. This should be a pickle file that holds the prepared and structured query data.
- `-e, --embedding_dir`: Directory containing the embeddings that represent the textual data of the destination texts.
- `-o, --output_path`: Path where the retrieval results will be stored. The output should be a JSON file. The format of the output is `{"query": {"city": (aggregate_score, {"aspect": (aspect_score, [top_chunks])})}}`, which provides scores and relevant text chunks for each city and query aspect.
- `--emb_type`: Specifies the type of embedder used to generate the embeddings. Available types are 'gpt' (GPT-based) or 'st' (Sentence Transformer-based). 

