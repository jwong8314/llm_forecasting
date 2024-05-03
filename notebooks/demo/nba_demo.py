# Standard library imports
import pickle

# Third-party library imports
import asyncio
import pandas as pd
# Local application/library specific imports
from config.constants import PROMPT_DICT
from utils.data_utils import get_formatted_data
from utils.visualize_utils import visualize_all, visualize_all_ensemble
import ranking
import summarize
import ensemble


# with open("sample_questions.pickle", "rb") as file:
#     sample_qs = pickle.load(file)
    
# with open("sample_questions.pickle", "wb") as file:
#     pickle.dump(sample_qs, file)
    
# formatted_data, raw_data = get_formatted_data(
#     "",
#     retrieval_index=1,
#     num_retrievals=5,
#     questions_after="2022",
#     return_raw_question_data=True,
#     data=sample_qs,
# )

# For this demo, we'll evaluate the first question.

import json
with open(f'../../../json/nba2.json') as f:
    arg_dict = json.load(f)

question = arg_dict['forecast_q']
background_info = arg_dict['background_info']
resolution_criteria = arg_dict['resolution_criteria']
df = pd.read_csv(f'../../../data/{arg_dict["data"]}')

answer = arg_dict['answer']
question_dates = ('2021-09-15', '2021-09-30')
retrieval_dates = ('2021-09-15', '2021-09-30')
urls_in_background = []






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
    "AGGREGATION_MODEL_NAME": "gpt-4",
    "AGGREGATION_WEIGTHTS": None,
}

today_to_close_date = [retrieval_dates[1], question_dates[1]]
ensemble_dict = asyncio.run(ensemble.meta_reason(
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
    weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
    meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
    meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
    meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
    dataframe = df
))
print(ensemble_dict['meta_prediction'])
ensemble_dict_v2 = asyncio.run(ensemble.meta_reason(
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
    weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
    meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
    meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
    meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
    dataframe = None
))
print("===== quant =====")
print(ensemble_dict['meta_prediction'])
print("===== qual =====")
print(ensemble_dict_v2['meta_prediction'])



