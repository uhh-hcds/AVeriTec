from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import random
import re
from helper import *


p = re.compile(r"(\d\.)? \'?(.+?)\'?$", re.MULTILINE)

def split_to_questions(df_train: dict):
    df1, df2, df3, df4 = [],[],[],[]
    for sample in df_train:
        if len(sample['questions'])==1:
            df1.append(sample)
        elif len(sample['questions'])==2:
            df2.append(sample)
        elif len(sample['questions'])==3:
            df3.append(sample)
        elif len(sample['questions'])==4:
            df4.append(sample)
    return df1, df2, df3, df4


def main(key: str, input_filepath_train: str, input_filepath_dev: str, output_filepath: str, model: str, num_regens: int):
    client = OpenAI(api_key=key)

    with open(input_filepath_train) as f:
        df_train = json.load(f)

    df1, df2, df3, df4 = split_to_questions(df_train)

    with open(input_filepath_dev) as f:
        df = json.load(f)

    output = {}
    for index in tqdm(range(len(df)), desc='Processing rows'):
        df_examples = [random.choice(df1), random.choice(df2), random.choice(df3), random.choice(df4)]
        content = f'''From the sentence below, please formulate up to 5 questions to help to do the fact checking. What do we need to know to check whether the claim is true? "Decompose" the claim into subquestions. Generate as few questions as possible.

Example:
'''     
        for example in df_examples:
            questions = str([q['question'] for q in example['questions']])[1:-1]
            content += f'''
claim: {example['claim']}
questions: {questions}

'''
        content+= f'''
claim: {df[index]['claim']}
questions:'''

        completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert summary reviewer."},
            {"role": "user", "content": content}
        ], n=num_regens
        )

        
        output[index] = df[index]
        output[index]['new_questions'] = [i[-1] for i in p.findall(completion.choices[0].message.content)]
    
    out_df = pd.DataFrame(output)
    out_df.to_json(output_filepath)


if __name__ == "__main__":
  main("sk", "AVeriTeC/data/train.json", 
  "test.json", "claims_with_questions_1-4_4_test.json", "gpt-4-turbo", 1)
