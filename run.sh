# parameters
embedder_type="st"
retrieve_type="dense"
mode="eqr_10"
k=5

# path
domain="hotel_nyc"
data_root_folder="data/${domain}"
query_path="${data_root_folder}/queries_20.txt"
ground_truth_path="${data_root_folder}/ground_truth_100_r8.json"
passage_dir="${data_root_folder}/corpus"
embedding_dir="${data_root_folder}/embeddings"

# path to save results
output_root_folder="output/${domain}/${mode}"

doc_chunks_dir="${embedding_dir}/chunks/section"
doc_embeddings_dir="${embedding_dir}/all-minilm-l6-v2/section"
output_query_path="${output_root_folder}/processed_query_${mode}.pkl"




# produce embeddings for the dataset
# python -m src.Embedder.embedderRunner -d $passage_dir -o $embedding_dir --emb_type $embedder_type

# run the query
python -m src.QueryProcessor.queryProcessorRunner --input_path $query_path --output_dir $output_root_folder --retriever_type $retrieve_type --mode $mode

# run the retriever
python -m src.Retriever.retrieverRunner --q $output_query_path --chunks_dir $doc_chunks_dir --embedding_dir $doc_embeddings_dir --retriever_type $retrieve_type --emb_type $embedder_type --output_dir $output_root_folder --num_chunks $k

# evaluate the results
ranked_list_path="${output_root_folder}/ranked_list.json"
echo "Running evaluation"
evaluator=("rprecision" "recall" "map")
for eval in "${evaluator[@]}"; do
    # if eval is rprecision, then k is not required
    if [ "$eval" == "rprecision" ]; then
        echo "Evaluating: $eval"
        python -m src.Evaluator.evaluatorRunner -e $eval -j $ranked_list_path -g $ground_truth_path -o ${output_root_folder}/results_${mode}/${eval}.json
    else
        for k in 10 30 50 100; do
            echo "Evaluating: $eval @ $k"
            python -m src.Evaluator.evaluatorRunner -e $eval -k $k -j $ranked_list_path -g $ground_truth_path -o ${output_root_folder}/results_${mode}/${eval}_at${k}.json
        done
    fi

done