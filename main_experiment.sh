# Define the mapping of "domain|embedder_name" to "k" values
declare -A k_values
k_values["hotel_nyc|all-MiniLM-L6-v2"]=50
k_values["hotel_nyc|msmarco-distilbert-base-tas-b"]=50
k_values["hotel_beijing|all-MiniLM-L6-v2"]=15
k_values["hotel_beijing|msmarco-distilbert-base-tas-b"]=40
k_values["travel_dest|all-MiniLM-L6-v2"]=30
k_values["travel_dest|msmarco-distilbert-base-tas-b"]=50
k_values["restaurant_phi|all-MiniLM-L6-v2"]=15
k_values["restaurant_phi|msmarco-distilbert-base-tas-b"]=15
k_values["restaurant_nol|all-MiniLM-L6-v2"]=8
k_values["restaurant_nol|msmarco-distilbert-base-tas-b"]=5

# Parameters
embedder_type="st"
retrieve_type="dense"

for domain in "hotel_beijing" "travel_dest" "restaurant_phi" "restaurant_nol"; do
# for domain in "hotel_nyc"; do

    for mode in "none" "q2e" "gqr" "q2d" "eqr_5" "eqr_8" "eqr_10" "eqr_12" "eqr_15"; do
    # for mode in "eqr_10"; do
        data_root_folder="data/${domain}"
        ground_truth_path="${data_root_folder}/ground_truth.json"
        query_path="${data_root_folder}/queries.txt"
        output_root_folder="output/${domain}/${mode}"
        
        ground_truth_path="${data_root_folder}/ground_truth.json"
        passage_dir="${data_root_folder}/corpus"

        # python -m src.QueryProcessor.queryProcessorRunner --input_path $query_path --output_dir $output_root_folder --retriever_type $retrieve_type --mode $mode

        for embedder_name in "all-MiniLM-L6-v2" "msmarco-distilbert-base-tas-b"; do
            # Fetch the associated k value for this domain and embedder
            key="${domain}|${embedder_name}"
            k=${k_values[$key]}


            embedding_dir="${data_root_folder}/embeddings/${embedder_name}"

            # Path to save results
            

            doc_chunks_dir="${embedding_dir}/chunks/section"
            doc_embeddings_dir="${embedding_dir}/section"
            output_query_path="${output_root_folder}/processed_query_${mode}.pkl"
            retrieval_output_folder="${output_root_folder}/${embedder_name}"

            # Produce embeddings for the dataset
            # python -m src.Embedder.embedderRunner -d $passage_dir -o $embedding_dir --emb_type $embedder_type --emb_name $embedder_name


            # Run the retriever
            retriever_output_dir="${retrieval_output_folder}_${k}"
            # python -m src.Retriever.retrieverRunner --q $output_query_path --chunks_dir $doc_chunks_dir --embedding_dir $doc_embeddings_dir --retriever_type $retrieve_type --emb_type $embedder_type --output_dir $retriever_output_dir --num_chunks $k --emb_name $embedder_name

            # Evaluate the results
            ranked_list_path="${retriever_output_dir}/ranked_list.json"
            echo "Running evaluation"
            evaluator=("rprecision" "recall" "map")
            for eval in "${evaluator[@]}"; do
                # If eval is rprecision, then k is not required
                if [ "$eval" == "rprecision" ]; then
                    echo "Evaluating: $eval"
                    python -m src.Evaluator.evaluatorRunner -e $eval -j $ranked_list_path -g $ground_truth_path -o $retriever_output_dir/results_${mode}/${eval}.json
                else
                    for k in 10 50; do
                        echo "Evaluating: $eval @ $k"
                        python -m src.Evaluator.evaluatorRunner -e $eval -k $k -j $ranked_list_path -g $ground_truth_path -o $retriever_output_dir/results_${mode}/${eval}_at${k}.json
                    done
                fi
            done
        done
    done
done


