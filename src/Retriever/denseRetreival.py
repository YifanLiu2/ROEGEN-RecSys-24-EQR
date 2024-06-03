import os
import pickle
import numpy as np
import tqdm

class denseRetrieval:
    def run_dense_retreival(percentile, queries, model, index_dir, save=True, output_dir=None, load_if_exists=True, path_suffix=None, query_embeds=None):
        full_path = os.path.join(output_dir, "query_dense_results_full.pkl")
        path = os.path.join(output_dir, f"query_dense_results_full_{path_suffix}.pkl") if path_suffix else full_path
        score_path = os.path.join(output_dir, f"query_dense_results_full_scores_{path_suffix}.pkl") if path_suffix else os.path.join(output_dir, "query_dense_results_full_scores.pkl")

        query_dense_results = {}
        pkls = sorted([f for f in os.listdir(index_dir)])

        skip = False
        query_scores = None

        if load_if_exists and os.path.exists(path):
        
            query_dense_results = pickle.load(open(path, "rb"))
            query_scores = pickle.load(open(score_path, "rb"))
            # load full path as well and combine
            if os.path.exists(full_path):
                query_dense_results_full = pickle.load(open(full_path, "rb"))
                for query in query_dense_results_full:
                    if query not in query_dense_results:
                        query_dense_results[query] = query_dense_results_full[query]
            skip = True
        
        percentile_thresholds = []
        query_scores_dict = {}
        for m, query in enumerate(queries):
            if skip and query in query_scores:
                percentile_thresholds.append(np.percentile(query_scores[query], percentile))
                continue
            query_scores_vals = []            
            if query_embeds is not None:
                query_emb = query_embeds[m]
            else:
                query_emb = model.encode([query])   
            for i in range(0, len(pkls), 2):
                emb_file = pkls[i]
                sent_file = pkls[i+1]
                city_name = emb_file.split("_")[0]
            
                emb = pickle.load(open(f"{index_dir}/{emb_file}", "rb"))
                curr_scores = (emb @ query_emb.T)
                query_scores_vals += curr_scores.tolist()
                
                del emb
            
            query_scores_vals = np.array(query_scores_vals)
            query_scores_dict[query] = query_scores_vals
            percentile_thresholds.append(np.percentile(query_scores_vals, percentile))
        
            if save:
                # save query scores
                with open(score_path, "wb") as f:
                    pickle.dump(query_scores_dict, f)

        del query_scores_dict
        
        if not skip or query not in query_dense_results:
            for j, query in tqdm(enumerate(queries)):
                dense_results = {}

                if query_embeds is not None:
                    query_emb = query_embeds[m]
                else:
                    query_emb = model.encode([query])  
                for i in range(0, len(pkls), 2):
                    emb_file = pkls[i]
                    sent_file = pkls[i+1]
                    city_name = emb_file.split("_")[0]

                    emb = pickle.load(open(f"{index_dir}/{emb_file}", "rb"))
                    sentences = pickle.load(open(f"{index_dir}/{sent_file}", "rb"))

                    # get score for each sentence
                    scores = []
                    for s in emb:
                        score = s @ query_emb.T
                        scores.append(float(score))

                    sent_score_tuples = list(zip(sentences, scores))
                    sent_score_tuples.sort(key=lambda x: x[1], reverse=True)

                    dense_results[city_name] = sent_score_tuples


                query_dense_results[query] = dense_results  

        if save:
            pickle.dump(query_dense_results, open(path, "wb"))
        
        # print(list(query_dense_results.keys()))
        
        out_query_dense_results = {}
        
        for j, query in enumerate(queries):
            out_dense_results = {}

            for i in range(0, len(pkls), 2):
                emb_file = pkls[i]
                sent_file = pkls[i+1]
                city_name = emb_file.split("_")[0]

                # get sentences above threshold
                top_sentences = []
                agg_score = 0
                for sent, score in query_dense_results[query][city_name]:
                    if score >= percentile_thresholds[j]:
                        top_sentences.append(sent)
                        agg_score += score
                
                out_dense_results[city_name] = (top_sentences, agg_score)

            sorted_dense_results = sorted(out_dense_results.items(), key=lambda x: x[1][1], reverse=True)

            out_query_dense_results[query] = sorted_dense_results
        
        del query_dense_results
        
        return out_query_dense_results