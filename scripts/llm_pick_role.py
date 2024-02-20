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
        
def process_batch(ques_df, template):    
    answer_prompts = []
    for idx, item in ques_df.iterrows():
        question_text = item['question']
        option1 = item["option1"]
        option2 = item["option2"]
        option3 = item["option3"]
        option4 = item["option4"]

        choices_text = f'Options: 1. {option1}, 2. {option2}, 3. {option3}, 4. {option4}.'
        question_text = f"{question_text} {choices_text}"
                 
        full_prompt = template.format(question=question_text) 
        
        answer_prompts.append(full_prompt)
        
    return answer_prompts


'''LOAD ARGS'''

args = parser.parse_args()
print(args)

use_cuda = args.use_cuda
batch_size = args.batch_size

'''WHOLE MMLU SAMPLE QUES'''
dataset_path = os.path.join(parent_dir, 'data/data_for_model_pick.csv')
question_df = pd.read_csv(dataset_path)

model_name_dic = {'flan-t5': ["google/flan-t5-xxl"], 'llama2': ["meta-llama/Llama-2-7b-chat"], 'opt': ["facebook/opt-iml-max-1.3b"]}

'''CHOOSE MODEL'''
model_name_lst = model_name_dic['llama2']
model = "llama2" 

'''CHOOSE TEMPLATE'''
template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a multiple-choice question. Select only the option number that corresponds to the correct answer for the following question. Choose a role that you think will yield the most accurate answer, state it, and then provide your answer in the format "Role: xxx; Answer: xxx."}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''

'''FOR FLAN MODEL PICK'''
# template = '''Select the most suitable role for providing an accurate and informed response to the following question. {question} Make sure to include both the role and the option number in the answer. Respond in the format: "Role; Answer" You MUST output the chosen role first.'''

# template = '''Select the most suitable role for providing an accurate and informed response to the following question. {question} Make sure to report both the ROLE and the OPTION NUMBER in the answer.'''

# template = '''Choose the best-suited role to respond to this question with accuracy and expertise. {question} Your answer should specify both the role and its option number. Reply in this manner: 'Role; Answer,' leading with the role.'''

# template = '''Select the most suitable role for providing an accurate and informed response to the following question. {question} Make sure to include both the role and the option number in the answer. Respond in the format: "Role; Answer"'''

# template = '''Select the most suitable role for providing an accurate and informed response to the following question. {question} Make sure to include both the role and the option number in the answer. Answer in the format: 'Role; Answer,' with the role mentioned first.'''


'''FOR FLAN CONTROL'''
# template = '''{question} Please select the correct answer number:'''

generator = init_pipeline(model_name_lst[0], use_cuda)
all_answers = []

for i in tqdm(range(0, len(question_df), batch_size)):
    batch_data = question_df[i:i+batch_size]
    batch_prompts = process_batch(batch_data, template)
    if re.search('flan', model):
        answers_raw = generator(batch_prompts, num_return_sequences=1)
    else:
        answers_raw = generator(batch_prompts, num_return_sequences=1,
                                pad_token_id=generator.tokenizer.eos_token_id,
                                max_new_tokens=30, repetition_penalty=1.1)

    for question_text, dataset, answer_output in zip(list(batch_data['full_question']), list(batch_data['dataset']), answers_raw):
        if re.search('flan', model):
            answer_text = answer_output["generated_text"].strip()
        else:
            answer_text = answer_output[0]["generated_text"].strip()
        question_dict = {"prompt": "Self-pick", "full_question": question_text, "dataset": dataset, "answer": answer_text}
        all_answers.append(question_dict)

answers_df = pd.DataFrame(all_answers)
output_path = os.path.join(parent_dir, 'data/mmlu_invariant_ans/llama_model_pick.csv')

answers_df.to_csv(output_path, index=False)











