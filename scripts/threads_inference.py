# import sys
# print(sys.executable)
import os
os.environ['TRANSFORMERS_CACHE'] = '/shared/3/cache/huggingface'
import argparse
import pandas as pd
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import tensorflow as tf
import torch
from tqdm import tqdm
import string, re, collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from datasets import load_dataset
import random
from sklearn.metrics import f1_score
from utilities import *
from inference import *
import datetime

from accelerate import Accelerator
from queue import Queue
import threading
from math import ceil


current_date = datetime.datetime.now()
formatted_date = current_date.strftime('%Y-%m-%d')

if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

torch.cuda.empty_cache()

parser = argparse.ArgumentParser("")
parser.add_argument("--model_type", type=str, default='huggingface')
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--batch_size_xxl", default=4, type=int)
parser.add_argument("--device_id", default=0, type=int)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--max_devices", type=int, default=None)
parser.add_argument("--devices", type=str, default=None)

def parse_devices(devices_str):
    if devices_str:
        return [int(device.strip()) for device in devices_str.split(',')]
    else:
        return None

device_model_dic = {}

def init_multi_gpu_queue(devices=None):
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    
    device_queue = Queue()
    for device_index in devices:
        if device_index < torch.cuda.device_count():
            device_queue.put(device_index)
        else:
            raise ValueError(f"Invalid device index {device_index}, only {torch.cuda.device_count()} devices available.")

    return device_queue

def init_threadsafe_generator(model_name, device_queue, lock, max_length=None):
    with lock:
        if device_queue.empty():
            return None
        device_id = device_queue.get()
        
    model = device_model_dic.get(device_id)
    if model is None:
        device_model_dic[device_id] = init_pipeline(model_name, True, device_id)
        model = device_model_dic[device_id]
    if max_length and re.search("llama", model_name):
        model.max_length = max_length
    return model

def init_pipeline(model_name, use_cuda, device_id):
    cache_dir='/shared/3/cache/huggingface'
    
    if re.search('llama', model_name):
        model_path = '/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'
        pipe_type = "text-generation"
        tokenizer_path = '/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'
        
        tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path, 
                                                   cache_dir=cache_dir, padding=True)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path, cache_dir=cache_dir)
    elif re.search('opt', model_name):
        pipe_type = "text-generation"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = "[PAD]"
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    elif re.search('flan', model_name):
        pipe_type = "text2text-generation"
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir, load_in_8bit=True)
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
    else:
        print("No model found!")

    if use_cuda:
        device = torch.device(f"cuda:{device_id}")
        print(f"Using gpu id: {device.index}")
        if re.search('opt', model_name):
            generator = pipeline(pipe_type, model=model, tokenizer=tokenizer, batch_size=batch_size, device=device.index)
        else:
            model.to(device)
            generator = pipeline(pipe_type, model=model, device=device.index, tokenizer=tokenizer, batch_size=batch_size)
    else:
        generator = pipeline(pipe_type, model=model, tokenizer=tokenizer)
            
    return generator 

def get_indefinite_article(word):
    if word[0].lower() in 'aeiou':
        return 'an'
    else:
        return 'a'

def process_batch(batch_data, role_prompt, role=None, indefinite=None):
    template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
    
    control_template = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a multiple-choice question. Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:'''
    
    flan_template = '''{role_context} {question} Please select the correct answer number:'''
    
    
    answer_prompts = []
    for idx, item in batch_data.iterrows():
        question_text = item['question']
        option1 = item["option1"]
        option2 = item["option2"]
        option3 = item["option3"]
        option4 = item["option4"]

        choices_text = f'Options: 1. {option1}, 2. {option2}, 3. {option3}, 4. {option4}.'
        question_text = f"{question_text} {choices_text}"
        if role_prompt is not None:
            try: 
                role_context = role_prompt.format(role=role, indefinite=indefinite)
            except:
                role_context = role_prompt.format(role=role)
                
            if re.search("t5", model_name):
                full_prompt = flan_template.format(role_context=role_context, question=question_text) 
            else:
                full_prompt = template.format(role_context=role_context, question=question_text)           
        else:
            full_prompt = control_template.format(question=question_text)
        answer_prompts.append(full_prompt)
        
    return answer_prompts

def generate_file_names(dataset, model, prompt_type, formatted_date, gpu_id=None):   
    if gpu_id is not None:
        suffix = f"_gpu_{gpu_id}"
    else:
        suffix = ""
        
    if prompt_type == "invariant":
        data_path = os.path.join(parent_dir, 'data', 'mmlu_invariant_ans/llama2-7b-chat-new')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_name = f"{dataset}_{model}_invariant_{formatted_date}{suffix}.json"
        
    else:
        data_path = os.path.join(parent_dir, 'data', 'mmlu_role_ans/opt-iml-max-1.3b/addon-new')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_name = f"{dataset}_{model}_role_{formatted_date}{suffix}.json"
                
    file_path = os.path.join(data_path, file_name) 
    
    save_finished_experiment(dataset, prompt_type, gpu_id) ## add gpu_id 
    
    return file_path

def multi_gpu_worker(device_queue, lock, all_answers, data, batch_size, model_name, current_roles, current_prompts, dataset_name, device_id, prompt_type):
    local_generator = init_threadsafe_generator(model_name, device_queue, lock)
    if local_generator is None:
        return

    local_all_answers = {}
    
    for role in tqdm(current_roles):
        indefinite = get_indefinite_article(role)
        print(role)
        for prompt in current_prompts:
            answers = []
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]
                answer_prompts = process_batch(batch_data, prompt,role, indefinite)
                
                if re.search("llama", model_name):
                    answer_outputs = local_generator(answer_prompts, num_return_sequences=1, 
                                                     pad_token_id=local_generator.tokenizer.eos_token_id, 
                                                     max_new_tokens=30, repetition_penalty=1.1)
                elif re.search("flan", model_name):
                    answer_outputs = local_generator(answer_prompts, num_return_sequences=1)
                else:
                    answer_outputs = local_generator(answer_prompts, num_return_sequences=1,
                                                     max_new_tokens=30, repetition_penalty=1.1)
                
                for full_prompt, question_text, answer_output in zip(answer_prompts, list(batch_data['question']), answer_outputs):
                    if re.search("flan", model_name):
                        answer_text = answer_output["generated_text"].strip()
                    else:
                        answer_text = answer_output[0]["generated_text"].strip()
                    question_dict = {"prompt": prompt, "full_prompt": full_prompt, "question": question_text, "answer": answer_text}
                    answers.append(question_dict)
                                                           
            local_all_answers[f"{role}_{prompt}"] = answers
            
        
    if re.search("llama", model_name):
        json_filename = generate_file_names(dataset_name, "llama", prompt_type, formatted_date, device_id)
    elif re.search("flan", model_name):
        json_filename = generate_file_names(dataset_name, "flan", prompt_type, formatted_date, device_id)
    else:
        json_filename = generate_file_names(dataset_name, "opt", prompt_type, formatted_date, device_id)
    
    with lock:
        all_answers.update(local_all_answers)
        print(f"--------------Writing results to file {json_filename}-------------")
        with open(json_filename, 'w') as f:
            json.dump(all_answers, f, indent=4)

            
'''MULTIPLE QUESTIONS'''

def get_invariant_prompts_ans(model_name, prompt_df, dataset_name, dataset_df, batch_size):
    num_devices = torch.cuda.device_count()
    num_device_to_use = min(num_devices, len(devices))
    print("Number of devices to use:", num_device_to_use)
    
    data = dataset_df[dataset_df.dataset == dataset_name]
    
    data_partitions = [data[i:i + ceil(len(data) / num_device_to_use)] for i in range(0, len(data), ceil(len(data) / num_device_to_use))]
    print("Length of data_partitions:", len(data_partitions)) 
    
#     current_prompts = prompt_df[prompt_df['category'] == 'non-context']['prompt']
    
    device_queue = init_multi_gpu_queue(devices)
    
    lock = threading.Lock()
    all_answers = {}
    
    threads = []
    num_threads = min(num_device_to_use, len(data_partitions))
    for i in range(num_threads):    
        partition = data_partitions[i]
        thread = threading.Thread(target=multi_gpu_worker, args=(device_queue, lock, all_answers, partition, batch_size, model_name, None, None, dataset_name, i, 'invariant'))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
        
    return all_answers


def get_role_prompts_ans(model_name, prompt_df, role_df, dataset_name, dataset_df, batch_size, use_cuda):
    
    num_devices = torch.cuda.device_count()
#     max_devices = args.max_devices
    num_device_to_use = min(num_devices, len(devices))
#     print("Number of available CUDAs:",num_devices)
    print("Number of devices to use:", num_device_to_use)
       
    # divide data into partitions 
    data = dataset_df[dataset_df.dataset == dataset_name]
    data_partitions = [data[i:i + ceil(len(data) / num_device_to_use)] for i in range(0, len(data), ceil(len(data) / num_device_to_use))]
    print("Length of data_partitions:", len(data_partitions)) 
    
    device_queue = init_multi_gpu_queue(devices)
    lock = threading.Lock()
    all_answers = {}
    threads = []
    
    current_roles = role_df['role']
    # get indefinite prompts 
    current_prompts = prompt_df['prompt']
   
    num_threads = min(num_device_to_use, len(data_partitions))
    for i in range(num_threads):    
        partition = data_partitions[i]
        thread = threading.Thread(target=multi_gpu_worker, args=(device_queue, lock, all_answers, partition, batch_size, model_name, current_roles, current_prompts, dataset_name, i, 'role'))
        thread.start()
        threads.append(thread)
        
    for thread in threads:
        thread.join()
        
    return all_answers




'''LOAD ARGS'''

args = parser.parse_args()
print(args)

use_cuda = args.use_cuda
model_type = args.model_type
batch_size = args.batch_size
devices = parse_devices(args.devices)

    
# device = torch.device("cuda" if use_cuda else "cpu")
# print("Device: ",device)

'''LOAD DATASETS'''  

'''TEST SET'''
dataset_path = os.path.join(parent_dir, 'data/test_set_questions_all.csv')
prompt_path = os.path.join(parent_dir, 'shared_data/invariant_prompts.csv')
role_path = os.path.join(parent_dir, 'shared_data/social-role-final.csv')
role_prompt_path = os.path.join(parent_dir, 'shared_data/role_prompts.csv')

all_dataset_df = pd.read_csv(dataset_path)

# prompt_df = pd.read_csv(prompt_path)

# prompts = ["Imagine you are talking to your {role}.",
#           "You are talking to your {role}.",
#           "Imagine you are talking to {indefinite} {role}.",
#           "You are talking to {indefinite} {role}.",
#           "You are {indefinite} {role}.",
#           "Imagine you are {indefinite} {role}."]

prompts = ["You are talking to your {role}.",
          "You are talking to {indefinite} {role}.",
          "You are {indefinite} {role}."]


role_prompt_df = pd.DataFrame({"prompt": prompts})

# role_prompt_df = pd.read_csv(role_prompt_path)
roles = ['human', 'AI', 'cat', 'God', 'person who knows everything', 'book', 'computer', 'desk', 'child', 'superhero', 'robot']

# roles = ['person who knows everything']
role_df = pd.DataFrame({"role": roles})

# role_df = pd.read_csv(role_path)

print(f'dataset df shape: {all_dataset_df.shape}')

dataset_lst = list(set(pd.read_csv(dataset_path)['dataset'].tolist()))

print(f'Task list: {dataset_lst}')

model_name_dic = {'flan-t5': ["google/flan-t5-xxl"], 'llama2': ["meta-llama/Llama-2-7b-chat"], 'opt': ["facebook/opt-iml-max-1.3b"]}

# model_sizes = ['xxl']
# model_name_lst = model_name_dic['flan-t5']

'''CHOOSE MODEL'''
model_name_lst = model_name_dic['opt']
model = "opt" 

prompt_types = ['role']

def save_finished_experiment(dataset, prompt_type, gpuid):
    try:
        with open('opt_addon_finished_experiments.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    data.append({
        "dataset": dataset,
        "prompt_type": prompt_type,
        "gpu_id": gpuid
    })

    with open('opt_addon_person_finished_experiments.json', 'w') as f:
        json.dump(data, f)
        
def is_experiment_done(dataset, prompt_type):
    try:
        with open('opt_addon_person_finished_experiments.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    for exp in data:
        if exp["dataset"] == dataset and exp["prompt_type"] == prompt_type:
            return True
    return False

                                          
##### multiple choice questions 

multi_dataset = dataset_lst


'''INFERENCE LOOP'''
for dataset_name in dataset_lst:
    for prompt_type in prompt_types:
        if is_experiment_done(dataset_name, prompt_type):
            print(f"Skipping experiment with dataset: {dataset_name}, prompt_type: {prompt_type}")
            continue 

        batch_size = args.batch_size
        for model_name in model_name_lst:
            torch.cuda.empty_cache()
            print(f'----Conducting Experiment: {dataset_name} & {model_name} & {prompt_type} & {batch_size}')

            if prompt_type == 'role':
                print("Generating role-prompted answers...")
                ans = get_role_prompts_ans(model_name, role_prompt_df, role_df, dataset_name, all_dataset_df, batch_size, use_cuda)
            else:
                print(f"Generating invariant prompted answers using {model_name}")
#                 ans = get_invariant_prompts_ans(model_name, prompt_df, dataset_name, all_dataset_df, batch_size)
                        
                 
                           
                            







