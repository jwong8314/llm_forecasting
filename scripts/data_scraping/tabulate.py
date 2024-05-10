import pickle
import json
from datetime import datetime
import pytz
tz = pytz.timezone('UTC')
import numpy as np

from utils.data_utils import get_formatted_data
today = datetime.now(tz = tz)
def filter_for_resolved():
    # Load the pickle file
    with open('./04_29/complete_dataset_04_29.pickle', 'rb') as file:
        data = pickle.load(file)
        
    resolved = [item for item in data if item['is_resolved']]
    with open ("./04_29/resolved_questions.pickle", "wb") as file:
        pickle.dump(resolved, file)
    # Count the number of closed questions
    closed_questions = len(resolved)
    total_questions = len(data)
    print(f"Number of closed questions: {closed_questions}")
    print(f"Number of total questions: {total_questions}")

def apply_scale(val,scale):
    if val >= 2 or val <= -1:
        return None
    if scale['deriv_ratio'] == 1:
        # linear scaling
        return scale['min'] + val * (scale['max'] - scale['min'])
    else: 
        ratio = (np.power(scale['deriv_ratio'], val) - 1) / (scale['deriv_ratio'] - 1)
        # log scaling
        return scale['min'] + ratio * (scale['max'] - scale['min'])
        

def compute_score():
    # open resolved_questions
    with open('./04_29/resolved_questions.pickle', 'rb') as file:
        resolved = pickle.load(file)
        resolved = [ r for r in resolved if r['resolution'] is not None ]
 
        resolved = [ r for r in resolved if r['close_time'] <  today and r['close_time'] > datetime(2023, 12, 31, tzinfo=tz)]
        print (len(resolved))
        import ipdb ; ipdb.set_trace()
        for s in resolved:
            if 'q1' not in s["community_prediction"]["unweighted"]:
                print (s["community_prediction"])
        samples = [ {
            "id": s['id'],
            "forecast_q": s['title'],
            "model": "gpt-4-1106-preview",
            "resolution_criteria": s['resolution_criteria'],
            "answer": apply_scale(s['resolution'],s['possibilities']['scale']),
            "background_info": s['background'],
            "question_dates": [
                s['close_time'].date().__str__(),
                s['close_time'].date().__str__()
            ],
            "linear_c": 0,
            "oom_c": 0, 
            "confidence": 0.5, 
            "scale": s['possibilities']['scale'],
            "raw_quartiles": {"q1": s['community_prediction']['unweighted']['q1'], "q3": s['community_prediction']['unweighted']['q3']} if len(s['community_prediction']['unweighted']['y']) > 0 else None,
            "quartiles": {"q1": apply_scale(s['community_prediction']['unweighted']['q1'], s['possibilities']['scale']),
                          "q3": apply_scale(s['community_prediction']['unweighted']['q3'], s['possibilities']['scale'])} 
            if len(s['community_prediction']['unweighted']['y']) > 0 else None,
            
        } for s in resolved
        ]
        import ipdb; ipdb.set_trace()
        for i, sample in enumerate(samples):
            json.dump(sample, open(f"./04_29/jsons/samples_{i}.json", "w"))
        




if __name__ == "__main__":
    compute_score()

