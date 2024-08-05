query_path="data/query.txt"
embedder_type="st"
# produce embeddings for the dataset
python -m src.Embedder.embedderRunner -d "data/wikivoyage_data_clean" -o "embeddings" --emb_type $embedder_type

# parameters
doc_chunks_dir="embeddings/chunks/section"
doc_embeddings_dir="embeddings/msmarco-distilbert-base-tas-b/section"
ground_truth_path="data/ground_truth/ground_truth.json"
output_root_folder="output"
retrieve_type="dense"
mode="eqr"
k="50"
ground_truth_path="data/ground_truth/ground_truth.json"

# run the query
python -m src.QueryProcessor.queryProcessorRunner --input_path $query_path --output_dir $output_root_folder --retrieve_type $retrieve_type --mode $mode --doc_chunks_dir $doc_chunks_dir --doc_embeddings_dir $doc_embeddings_dir --ground_truth_path $ground_truth_path --k "18"

output_query_path="output/processed_query_${mode}.pkl"

# run the retriever
python -m src.Retriever.retrieverRunner --q $output_query_path --chunks_dir $doc_chunks_dir --embeddings_dir $doc_embeddings_dir --retrieve_type $retrieve_type --emb_type $embedder_type --k $k --output_dir $output_root_folder

# evaluate the results
ranked_list_path="${output_root_folder}/ranked_list.json"
echo "Running evaluation"
evaluator=("rprecision" "recall" "map")
for eval in "${evaluator[@]}"; do
    # if eval is rprecision, then k is not required
    if [ "$eval" == "rprecision" ]; then
        echo "Evaluating: $eval"
        python -m src.Evaluator.evaluatorRunner -e $eval -j $ranked_list_path -g $ground_truth_path -o ${retriever_output_dir}/results/${eval}.json
    else
        for k in 30 50 100; do
            echo "Evaluating: $eval @ $k"
            python -m src.Evaluator.evaluatorRunner -e $eval -k $k -j $ranked_list_path -g $ground_truth_path -o ${retriever_output_dir}/results/${eval}_at${k}.json
        done
    fi

done