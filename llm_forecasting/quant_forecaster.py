from openai import OpenAI
import pandas as pd
import re
import model_eval

def quant_forecaster( df, forecast_question, background, resolution_criteria, model_name = "gpt-4-turbo", date_description='2020-2021, so you have no access to data from 2022', df_description=''):
    """
    Perform quantitative forecasting based on a given question and a pandas dataframe.

    Args:
        api_key (str): The API key for accessing the OpenAI service.
        df (pandas.DataFrame): The historical dataframe used for forecasting.
        forecast_question (str): The question to be answered through forecasting.
        description (str, optional): Description of the dataframe. Defaults to '2020-2021, so you have no access to data from 2022'.

    Returns:
        None
    """
    



    example_row = 'The first row of the dataframe has '
    for c, v in zip(df.loc[0].index, df.loc[0].values):
        example_row += str(v) + ' for the ' + str(c) + ' column, '
        
    br_msg_sys = ("You are a helpful AI specialized in quantitative forecasting answering "+
            f"the question: {forecast_question}? You are provided a python pandas historical ({date_description}) dataframe with "+
            f"the columns being: {str(df.columns)} and the corresponding columns types: {str(df.dtypes)}. For example, {example_row}. ")
    br_msg_usr = ("Generate three (3) questions that can be simply answered with pandas code applied to "+
            f"the dataframe provided, denoted as 'df', that also can answer the forecasting question. The dataframe comes with the description:\n {df_description}\n These questions should result ONLY in numerical values. Make a very clear note of what each row of the dataframe is, and make sure your questions answer the provided question given what each row AND column represent."+
            "Only provide the questions, and NOT the code. Do not use vague phrases like specific subset, "+
            "specific sub/target category, or some value. You must be specific and make judgement calls "+
            """on thresholds, ranges, or values. """ + 
"""
Here are three types of helpful questions: 
1. Base rates historically: questions that ask about the average over the data available in the dataset.
2. Recent trends: questions that ask about recent values in the dataset.
3. Extreme values: questions about outlier values (min/max) historically can help with estimating the range of values.
""" +
            "Wrap the questions in ** and ** like: **<insert question here>**.")
            
    br_llm = model_eval.get_response_from_model(
        model_name=model_name,
        prompt=br_msg_usr,
        system_prompt=br_msg_sys,
    )  # raw response
    br_questions = br_llm
    br_questions= re.findall(r"\*\*(.*?)\*\*", br_questions) 
    #ipdb.set_trace()
    br_answers = []
    #print(f'BR QUESTIONS: {br_questions}')
    df_copy = df.copy()
    for q in br_questions:
        df = df_copy.copy()
        df_msg_sys = ("You are a helpful AI specialized in code generating and writing "+
                "based on provided questions regarding a dataframe. For example, if asked to compute the mean of " +
                "of column X, a good response would be df['X'].mean(). You are provided a python pandas dataframe named 'df' with "+
            f"the columns being: {str(df.columns)} and the corresponding columns types: {str(df.dtypes)}. For example, {example_row}.  "+
                "is named 'df'. Do not include any comments, only the code itself. For example, for the following question: Question 2: What is the average points per game scored by a specific player, for example, Player 'LeBron James' in the dataset?  " + 
    """```python
    df[df['Player'] == 'James,LeBron']['Pts'].mean()
    ```
    Start all code with ```python\n and end all code with \n```. Assume """+
                "that the code you generate will be directly inputted into an 'eval' function to be run, so it must be correct. Additionally, make sure the code outputs a number. ")
        
        df_msg_usr = f"Generate single line/one-liner code for the following question: {q}."
        df_llm = model_eval.get_response_from_model(
            model_name=model_name,
            prompt=df_msg_usr,
            system_prompt=df_msg_sys,
        )
        #print(df_msg)
        #print('-------;')
        q_response = df_llm
        #print(q_response)
        #print(q_response.split("```python\n"))
        #q_response = q_response.split("```python")[1].split("\n```")[0]
        #print(re.findall(pattern, q_response))
        print(q_response)
        q_response = re.findall(r"\`\`\`python\n(.*?)\n\`\`\`", q_response, re.DOTALL)[0].strip() #.split("```python")[1].split("\n```")[0]
        #print('----')
        #print(f' output value: {eval(q_response)}')
        operations = q_response.replace("\n", ";").split(";")
        res = None
        for operation in operations:
            try :
                res = eval(operation)
            except Exception as e:
                res = str(e)
                
        print (res)
        br_answers.append(res)

    base_reasonings = list(zip(br_questions, br_answers))
    base_reasonings_str = "\n".join([f"\nQuestion: {q}\nAnswer: {a}" for q, a in base_reasonings])

    forecast_msg_sys = """You are an expert superforecaster, familiar with the work of Tetlock and others.
    Your mission is to generate accurate predictions for forecasting questions.
    Aggregate the information provided by the user. Make sure to give detailed reasonings."""
    forecast_msg_usr = (f"I need your assistance with making a forecast. Here is the question and its metadata. \n\nQuestion: {forecast_question}\n\nBackground: {background}\n\nResolution criteria: {resolution_criteria}"+
                f"\nIn addition, I have generated a collection of other responses and reasonings from other forecasters, below are some of the common questions and answers: {base_reasonings_str} \nYour goal is to aggregate the information and make a final prediction."+
                """Instructions:
    1. Provide reasons why the answer might be high.
    {{ Insert your thoughts here }}

    2. Provide reasons why the answer might be low.
    {{ Insert your thoughts here }}

    3. Aggregate your considerations.
    {{ Insert your aggregated considerations here }}

    4. Output your prediction (an 80 percent confidence in the format [lowerbound, upperbound]) with an asterisk at the beginning and end of the confidence interval (e.g. '*[1.8,5.3]*').
    {{ Insert the confidence interval here }}""")
    ff_llm = model_eval.get_response_from_model(
        model_name=model_name,
        prompt=forecast_msg_usr,
        system_prompt=forecast_msg_sys,
    )
    quant_questions = f"""I have generated a collection of other responses and reasonings from other forecasters, below are some of the common questions and answers: {base_reasonings_str}\n\n"""
    print (ff_llm)

    return quant_questions + ff_llm
if __name__ == "__main__":
    import json
    with open(f'../../../json/nba2.json') as f:
        arg_dict = json.load(f)

    forecast_question = arg_dict['forecast_q']
    df = pd.read_csv(f'../../data/{arg_dict["data"]}')
    out = quant_forecaster(df, forecast_question, arg_dict['background'], arg_dict['resolution_criterion'], model_name = arg_dict['model'], description=arg_dict['description'])
    print (out)