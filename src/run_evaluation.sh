# ranked_list_path="output\new_test\results\rank_list_geqe.json"
# ground_truth_path="data\ground_truth\ground_truth.json"
# retriever_output_dir="output\new_test\results\geqe"
ranked_list_path="output\best_results\results\v_2\rank_list_geqe.json"
ground_truth_path="data\ground_truth\ground_truth.json"
retriever_output_dir="output\best_results\results\v_2\geqe"
echo "Evaluating the retriever output"
evaluator=("precision" "rprecision" "recall" "map")
for eval in "${evaluator[@]}"; do
    # if eval is rprecision, then k is not required
    if [ "$eval" == "rprecision" ]; then
        echo "Evaluating: $eval"
        python -m src.Evaluator.evaluatorRunner -e $eval -j $ranked_list_path -g $ground_truth_path -o ${retriever_output_dir}/evaluator_results/${query_processor_mode}_${eval}.json
    else
        for k in 10 30 50; do
            echo "Evaluating: $eval @ $k"
            python -m src.Evaluator.evaluatorRunner -e $eval -k $k -j $ranked_list_path -g $ground_truth_path -o ${retriever_output_dir}/evaluator_results/${query_processor_mode}_${eval}_at${k}.json
        done
    fi
 
done

echo "Making tables"
python -m src.Evaluator.viewResults ${retriever_output_dir}/evaluator_results