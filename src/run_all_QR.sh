#!/bin/bash
echo "Current directory: $(pwd)"
original_query_input_path="data/final_queries.txt"

embedder_type="st"
doc_chunks_dir="embeddings/chunks/section"
doc_embeddings_dir="embeddings/paraphrase-minilm-l6-v2/section"
ground_truth_path="data/ground_truth/ground_truth.json"

output_root_folder="output/QR_test"
# run 10 times
for i in 1; do
    # query processor for none
    modes=("genqr")
    for query_processor_mode in "${modes[@]}"; do
        echo "Running query processor: "$query_processor_mode
        python -m src.QueryProcessor.quesryProcessorRunner --input_path $original_query_input_path --o $output_root_folder --mode $query_processor_mode
        processed_query_output_path="${output_root_folder}/processed_query_${query_processor_mode}.pkl"
        retriever_output_dir="${output_root_folder}/${embedder_type}_${query_processor_mode}_${i}"
        # run dense retriever
        python -m src.Retriever.retrieverRunner --q $processed_query_output_path --o $retriever_output_dir --chunks_dir $doc_chunks_dir --embedding_dir $doc_embeddings_dir --emb_type $embedder_type -nb "12" -na "7" --power "1"
        ranked_list_path="${retriever_output_dir}/ranked_list.json"
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
            echo "Making tables"
            python -m src.Evaluator.makeCSV --o ${retriever_output_dir}/results
    done
done

echo "Combine all results"
python -m src.Evaluator.combineCSV --o $output_root_folder

