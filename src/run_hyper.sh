#!/bin/bash

# original query input path

original_query_input_path="data/general_queries.txt"

embedder_type="st"
doc_chunks_dir="embeddings/chunks/section"
doc_embeddings_dir="embeddings/paraphrase-minilm-l6-v2/section"
ground_truth_path="data/ground_truth/ground_truth.json"

# query processor for none
output_root_folder="output/hyper_test"
# python -m src.QueryProcessor.queryProcessorRunner --input_path $original_query_input_path --o $output_root_folder

# retriever runner for none
# hyper parameters:
# num_chunks_broad: 1 to 20

echo "Running retriever for none"
for num_chunks_broad in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    output_save_folder="$output_root_folder/b${num_chunks_broad}"
    python -m src.Retriever.retrieverRunner --q $output_root_folder/processed_query_none.pkl --chunks_dir $doc_chunks_dir --embedding_dir $doc_embeddings_dir --output_dir $output_save_folder --emb_type $embedder_type --n $num_chunks_broad
    ranked_list_path="${output_save_folder}/ranked_list.json"
    echo "Running evaluation"
    evaluator=("rprecision" "recall" "map")
    for eval in "${evaluator[@]}"; do
        # if eval is rprecision, then k is not required
        if [ "$eval" == "rprecision" ]; then
            echo "Evaluating: $eval"
            python -m src.Evaluator.evaluatorRunner -e $eval -j $ranked_list_path -g $ground_truth_path -o ${output_save_folder}/results/${eval}.json
        else
            for k in 30 50 100; do
                echo "Evaluating: $eval @ $k"
                python -m src.Evaluator.evaluatorRunner -e $eval -k $k -j $ranked_list_path -g $ground_truth_path -o ${output_save_folder}/results/${eval}_at${k}.json
            done
        fi
    
    done
    echo "Making tables"
    python -m src.Evaluator.makeCSV --o ${output_save_folder}/results
done

echo "Combine all results"
python -m src.Evaluator.combineCSV --o $output_root_folder