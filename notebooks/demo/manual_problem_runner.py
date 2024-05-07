# Standard library imports
import pickle

# Third-party library imports
import asyncio
import pandas as pd
# Local application/library specific imports
from config.constants import PROMPT_DICT
from utils.data_utils import get_formatted_data
from utils.visualize_utils import visualize_all, visualize_all_ensemble
from utils.metrics_utils import s0_linear, s0_oom

import ranking
import summarize
import ensemble

from pathlib import Path
import json
import fire
import shutil

def eval_forecast(log_dir:str,  path_to_config:str, data_dir:str):
    # path_to_config = f'../../../json/nba2.json'
    with open(path_to_config, "r") as f:
        arg_dict = json.load(f)

    question = arg_dict['forecast_q']
    background_info = arg_dict['background_info']
    resolution_criteria = arg_dict['resolution_criteria']
    df = pd.read_csv(Path(data_dir) / f'{arg_dict["data"]}')

    answer = arg_dict['answer']
    question_dates = arg_dict['question_dates']
    retrieval_dates = question_dates
    urls_in_background = []



    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    shutil.copy(path_to_config, log_dir / "problem.json")

    RETRIEVAL_CONFIG = {
        "NUM_SEARCH_QUERY_KEYWORDS": 3,
        "MAX_WORDS_NEWSCATCHER": 5,
        "MAX_WORDS_GNEWS": 8,
        "SEARCH_QUERY_MODEL_NAME": "gpt-4-1106-preview",
        "SEARCH_QUERY_TEMPERATURE": 0.0,
        "SEARCH_QUERY_PROMPT_TEMPLATES": [
            PROMPT_DICT["search_query"]["0"],
            PROMPT_DICT["search_query"]["1"],
        ],
        "NUM_ARTICLES_PER_QUERY": 5,
        "SUMMARIZATION_MODEL_NAME": "gpt-3.5-turbo-1106",
        "SUMMARIZATION_TEMPERATURE": 0.2,
        "SUMMARIZATION_PROMPT_TEMPLATE": PROMPT_DICT["summarization"]["9"],
        "NUM_SUMMARIES_THRESHOLD": 10,
        "PRE_FILTER_WITH_EMBEDDING": True,
        "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
        "RANKING_MODEL_NAME": "gpt-3.5-turbo-1106",
        "RANKING_TEMPERATURE": 0.0,
        "RANKING_PROMPT_TEMPLATE": PROMPT_DICT["ranking"]["0"],
        "RANKING_RELEVANCE_THRESHOLD": 4,
        "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
        "SORT_BY": "date",
        "RANKING_METHOD": "llm-rating",
        "RANKING_METHOD_LLM": "title_250_tokens",
        "NUM_SUMMARIES_THRESHOLD": 20,
        "EXTRACT_BACKGROUND_URLS": True,
    }
    
    REASONING_CONFIG = {
        "BASE_REASONING_MODEL_NAMES": ["gpt-4-1106-preview", "gpt-4-1106-preview"],
        "BASE_REASONING_TEMPERATURE": 1.0,
        "BASE_REASONING_PROMPT_TEMPLATES": [
            [
                PROMPT_DICT["binary"]["scratch_pad"]["quant_1"],
                PROMPT_DICT["binary"]["scratch_pad"]["quant_2"],
            ],
            [
                PROMPT_DICT["binary"]["scratch_pad"]["quant_3"],
                PROMPT_DICT["binary"]["scratch_pad"]["quant_6"],
            ],
        ],
        "ALIGNMENT_MODEL_NAME": "gpt-4-turbo-preview",
        "ALIGNMENT_TEMPERATURE": 0,
        "ALIGNMENT_PROMPT": PROMPT_DICT["alignment"]["0"],
        "AGGREGATION_METHOD": "meta",
        "AGGREGATION_PROMPT_TEMPLATE": PROMPT_DICT["meta_reasoning"]["quant_0"],
        "AGGREGATION_TEMPERATURE": 0.2,
        "AGGREGATION_MODEL_NAME": "gpt-4-1106-preview",
        "AGGREGATION_WEIGHTS": None,
    }
    
    with open (log_dir / "configs.json", "w") as f:
        json.dump({"RETRIEVAL_CONFIG": RETRIEVAL_CONFIG, "REASONING_CONFIG": REASONING_CONFIG}, f)
    if not (log_dir / "all_summaries.json").exists():
        ranked_articles,all_articles,search_queries_list_gnews, search_queries_list_nc = asyncio.run(ranking.retrieve_summarize_and_rank_articles( question,
            background_info,
            resolution_criteria,
            retrieval_dates,
            urls=urls_in_background,
            config=RETRIEVAL_CONFIG,
            return_intermediates=True,
        ))

        all_summaries = summarize.concat_summaries(
            ranked_articles[: RETRIEVAL_CONFIG["NUM_SUMMARIES_THRESHOLD"]]
        )

        print(all_summaries[:3000], "...")
        
        with open (log_dir / "all_summaries.json", "w") as f:
            json.dump(all_summaries, f)
    else:
        with open (log_dir / "all_summaries.json", "r") as f:
            all_summaries = json.load(f)

    today_to_close_date = [retrieval_dates[1], question_dates[1]] 
    if not (log_dir / "ensemble_dict_with_data.json").exists():
        ensemble_dict_with_data = asyncio.run(ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
            base_model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
            base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
            aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
            answer_type="confidence_interval",
            weights=REASONING_CONFIG["AGGREGATION_WEIGHTS"],
            meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
            dataframe = df, 
            df_date_description = arg_dict['df_date_description'], 
            df_description = arg_dict['description']
        ))
        print(ensemble_dict_with_data['meta_prediction'])
        
        with open(log_dir / "ensemble_dict_with_data.json", "w") as f:
            json.dump(ensemble_dict_with_data, f)
    else:
        with open (log_dir / "ensemble_dict_with_data.json", "r") as f:
            ensemble_dict_with_data = json.load(f)
    
    if not (log_dir / "ensemble_dict_baseline.json").exists():
        ensemble_dict_baseline = asyncio.run(ensemble.meta_reason(
            question=question,
            background_info=background_info,
            resolution_criteria=resolution_criteria,
            today_to_close_date_range=today_to_close_date,
            retrieved_info=all_summaries,
            reasoning_prompt_templates=REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
            base_model_names=REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"],
            base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
            aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
            answer_type="confidence_interval",
            weights=REASONING_CONFIG["AGGREGATION_WEIGHTS"],
            meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
            meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
            meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
            dataframe = None
        ))
        
        with open(log_dir / "ensemble_dict_baseline.json", "w") as f:
            json.dump(ensemble_dict_baseline, f)
    else:
        with open(log_dir / "ensemble_dict_baseline.json", "r") as f:
            ensemble_dict_baseline = json.load(f)
    
    print("===== quant =====")
    print(ensemble_dict_with_data['meta_prediction'])
    print("===== qual =====")
    print(ensemble_dict_baseline['meta_prediction'])
    print("===== answer =====")
    print(answer)
    # metrics = {
    #     "baseline": {
    #         "s0_linear": s0_linear(ensemble_dict_baseline['meta_prediction'], answer),
    #         "s0_oom": s0_oom(ensemble_dict_baseline['meta_prediction'], answer)
    #     },
    #     "with_data": {
    #         "s0_linear": s0_linear(ensemble_dict_with_data['meta_prediction'], answer),
    #         "s0_oom": s0_oom(ensemble_dict_with_data['meta_prediction'], answer)
    #     }
    # }
    
    # with open(log_dir / "metrics.json", "w") as f:
    #     json.dump(metrics, f)
        
    # print("===== metrics =====")
    
    # for exp_case, metric_result in metrics.items():
    #     print(f"\t===== {exp_case} =====")
    #     for metric_name, metric_value in metric_result.items():
    #         print(f"\t{metric_name}: {metric_value}")
    
    


if __name__ == "__main__":
    fire.Fire(eval_forecast)

