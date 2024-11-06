from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import random
import re
from helper import *


p = re.compile(r"(\d\.)? \'?(.+?)\'?$", re.MULTILINE)

def main(key: str, input_filepath_train: str, input_filepath_dev: str, output_filepath: str, model: str, num_regens: int):
    client = OpenAI(api_key=key)

    with open(input_filepath_train) as f:
        df_train = json.load(f)

    df = []
    with open(input_filepath_dev) as f:
        for line in f:
            df.append(json.loads(line))


    output = {}
    for index in tqdm(range(len(df)), total=len(df), desc='Processing rows'):
        print(df[index]['claim_id'])
        for index_asnwer in tqdm(df[index]['top_10'][:50], total=50):
            df_examples = random.sample(df_train, k=3)
            content = f'''From the sentence below, please formulate 1 question that could be answered with this question. This question and answer should help to do the fact checking for the claim that is also given. Which question would be asked to get this asnwer given that we need to know whether the claim is true?

Examples:
'''     
            for example in df_examples:
                content += f'''
claim: {example['claim']}
'''
                for question in example['questions']:
                    content += f'''
answer: {question['answers'][0]['answer']}
question: {question['question']}
'''
            content += f'''
claim: {df[index]['claim']}

answer: {index_asnwer['sentence']}
question:'''

            completion = client.chat.completions.create(
                model=model,
                messages=[
                {"role": "system", "content": "You are an expert detective."},
                {"role": "user", "content": content}
            ], n=num_regens
            )

            index_asnwer['question'] = completion.choices[0].message.content
        
        output[index] = df[index]
        
    out_df = pd.DataFrame(output)
    out_df.to_json(output_filepath)


if __name__ == "__main__":
  main("sk", "AVeriTeC/data/train.json", 
  "UHH-AVERITEC/bm25_vectors_dev_top_10_unique_reproduce.json", "bm25_vectors_dev_top_10_unique_reproduce_with_questions.json", "gpt-4o-mini", 1)
