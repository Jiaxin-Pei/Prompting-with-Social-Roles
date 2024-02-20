import os
os.environ['TRANSFORMERS_CACHE'] = '/shared/3/cache/huggingface'
import json
import csv
import pandas as pd
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import tensorflow as tf
from tqdm import tqdm
import string, re, collections
import datetime

current_date = datetime.datetime.now()
formatted_date = current_date.strftime('%Y-%m-%d')

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

torch.cuda.empty_cache()

parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--use_cuda", type=bool, default=True)

def init_pipeline(model_name, use_cuda):
    cache_dir='/shared/3/cache/huggingface'
    
    if re.search('llama', model_name):
        model_path = '/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'
        pipe_type = "text-generation"
        tokenizer_path = '/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path, 
                                                   cache_dir=cache_dir, padding=True)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, 
                                                 cache_dir=cache_dir, device_map="auto")
    elif re.search('opt', model_name):
        pipe_type = "text-generation"
    elif re.search('flan', model_name):
        pipe_type = "text2text-generation"
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, load_in_8bit=True)
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir,device_map="auto")
    else:
        print("No model found!")

    if use_cuda:
        if re.search('opt', model_name):
            generator = pipeline(pipe_type, model=model_name, batch_size=batch_size, device_map="auto")
        else:
            generator = pipeline(pipe_type, model=model, tokenizer=tokenizer, 
                                 batch_size=batch_size, device_map="auto")
    else:
        generator = pipeline(pipe_type, model=model, tokenizer=tokenizer)
            
    return generator 
        
def process_batch(ques_df, model_name, context_prompt, role=None, indefinite=None):
    template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
    
    flan_template = '''{role_context} {question} Please select the correct answer number:'''
    
    answer_prompts = []
    for idx, item in ques_df.iterrows():
        question_text = item['question']
        option1 = item["option1"]
        option2 = item["option2"]
        option3 = item["option3"]
        option4 = item["option4"]

        choices_text = f'Options: 1. {option1}, 2. {option2}, 3. {option3}, 4. {option4}.'
        question_text = f"{question_text} {choices_text}"
        if context_prompt is not None:
            try:
                role_context = context_prompt.format(role=role, indefinite=indefinite)
            except:
                role_context = context_prompt.format(role=role)
            if re.search("t5", model_name):
                full_prompt = flan_template.format(role_context=role_context, question=question_text) 
            else:
                full_prompt = template.format(role_context=role_context, question=question_text)
        else:
            full_prompt = control_template.format(question=question_text)
        
        answer_prompts.append(full_prompt)
        
    return answer_prompts


def get_indefinite_article(word):
    if word[0].lower() in 'aeiou':
        return 'an'
    else:
        return 'a'
    
'''LOAD ARGS'''

args = parser.parse_args()
print(args)

use_cuda = args.use_cuda
batch_size = args.batch_size

'''LOAD QUESTIONS'''
dataset_path = os.path.join(parent_dir, 'data/final_mmlu_sample_ques.csv')
question_df = pd.read_csv(dataset_path)

'''ROLES & PROMPTS'''
### load lists of prompts and roles 
prompts = ["Imagine you are talking to your {role}.",
          "You are talking to your {role}.",
          "Imagine you are talking to {indefinite} {role}.",
          "You are talking to {indefinite} {role}.",
          "You are {indefinite} {role}.",
          "Imagine you are {indefinite} {role}."]

roles = ['person who knows everything']

model_name_dic = {'flan-t5': ["google/flan-t5-xxl"], 'llama2': ["meta-llama/Llama-2-7b-chat"], 'opt': ["facebook/opt-iml-max-1.3b"]}

'''CHOOSE MODEL'''
model_name_lst = model_name_dic['flan-t5']
model = "flan-t5" 
model_name = model_name_lst[0]
generator = init_pipeline(model_name_lst[0], use_cuda)

dataset_lst = list(set(pd.read_csv(dataset_path)['dataset'].tolist()))

for dataset in dataset_lst:
    data = question_df[question_df['dataset'] == dataset]
    print(f"-----Processing dataset {dataset}-----")
    all_answers = []
    for role in tqdm(roles):
        indefinite = get_indefinite_article(role)
        for prompt in prompts:
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]
                batch_prompts = process_batch(batch_data, model_name, prompt, role, indefinite)    
                if re.search('flan', model):
                    answers_raw = generator(batch_prompts, num_return_sequences=1)
                elif re.search('llama', model):
                    answers_raw = generator(batch_prompts, num_return_sequences=1,
                                            pad_token_id=generator.tokenizer.eos_token_id,
                                            max_new_tokens=30, repetition_penalty=1.1)
                else:
                    answer_outputs = local_generator(answer_prompts, num_return_sequences=1,
                                                         max_new_tokens=30, repetition_penalty=1.1)

                for question_text, dataset, answer_output in zip(list(batch_data['question']), list(batch_data['dataset']), answers_raw):
                    if re.search('flan', model):
                        answer_text = answer_output["generated_text"].strip()
                    else:
                        answer_text = answer_output[0]["generated_text"].strip()
                    question_dict = {"prompt": prompt, "question": question_text, "dataset": dataset, 
                                     "role": role, "answer": answer_text}
                    all_answers.append(question_dict)

    answers_df = pd.DataFrame(all_answers)
    
file_name = f"data/{model}_personwhoknowseverything.csv"
output_path = os.path.join(parent_dir, file_name)
answers_df.to_csv(output_path, index=False)











