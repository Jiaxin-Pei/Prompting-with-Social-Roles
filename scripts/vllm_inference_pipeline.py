import os
os.environ['HF_HOME'] = '/shared/3/cache/huggingface'
import json
import csv
import pandas as pd
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import string, re, collections
import datetime

from vllm import LLM, SamplingParams
from huggingface_hub import login

def get_key_for_value(dic, value):
    for key, values in dic.items():
        if value in values:
            return key
    return None

def format_choices(item):
    options = [item[f"option{i}"] for i in range(1, 5)]
    choices_text = ', '.join(f'{i}. {option}' for i, option in enumerate(options, 1))
    return f"{item['question']} Options: {choices_text}"

def init_pipeline(model_name, model_name_dic, tensor_parallel_size, download_dir):
    # Map model to its family name
    model_type = get_key_for_value(model_name_dic, model_name) 
    
    if model_type in model_name_dic: 
        print(f"Model Type is {model_type}")
        generator = LLM(model=model_name, 
                        download_dir=download_dir, 
                        tensor_parallel_size=tensor_parallel_size, 
                        enforce_eager=True)
    else:
        print(f"Model '{model_name}' not supported!")
        return None 
        
    return generator 

def process_prompts(ques_df, model_type, context_prompt=None, role=None, indefinite=None):
    
    prompt_templates = {
        'default': '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{{You will be presented with a role-playing context followed by a multiple-choice question. {role_context} Select only the option number that corresponds to the correct answer for the following question.}}\n\n### Input:\n{{{{{question}}} Provide the number of the correct option without explaining your reasoning.}} \n\n### Response:''',
        'llama3': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{role_context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe following is a multiple choice question (with answers). Reply with only the option number. {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
        'flan': '{role_context} {question} Please select the correct answer number:',
        'mistral': '<s>[INST]{role_context} The following is a multiple choice question (with answers). Reply with only the option number.\n {question}</s>[/INST]',
        'qwen': '<|system|>\n{role_context} The following is a multiple choice question (with answers). Reply with only the option number.</s> \n<|user|>\n{question}</s>\n<|assistant|>\n',
        'qwen2.5': '<|im_start|>system\\n{role_context} The following is a multiple choice question (with answers). Reply with only the option number.<|im_end|>\\n<|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n'
    }
    
    answer_prompts = []
    for _, item in ques_df.iterrows():
        question_text = format_choices(item)
        
        role_context = ""
        if context_prompt:
            try:
                role_context = context_prompt.format(role=role, indefinite=indefinite)
            except KeyError:
                role_context = context_prompt.format(role=role)
        else:
            role_context = ""
            
        template_key = model_type if model_type in prompt_templates else 'default'       
        full_prompt = prompt_templates[template_key].format(role_context=role_context, question=question_text)
       
        answer_prompts.append(full_prompt)
        
    return answer_prompts

def get_indefinite_article(word):
    if word[0].lower() in 'aeiou':
        return 'an'
    else:
        return 'a'

def generate_answers(data, prompts_lst, model, model_name_dic, generator, sampling_params):
    """Generates answers and returns a list of dictionaries containing question details."""
    all_answers = []
    
    if generator:
        answers_raw = generator.generate(prompts_lst, sampling_params, use_tqdm=False)
    else:
        print("Model not supported!")
        return all_answers  # Return empty list if model not supported

    for true_option, groundtruth, question_id, dataset, answer_output in zip(data['true_option'], 
                                                                             data['groundtruth'], 
                                                                             data['question_id'], 
                                                                             data['dataset'], 
                                                                             answers_raw):
        answer_text = answer_output.outputs[0].text
        question_dict = {
            "dataset": dataset,
            "true_option": true_option,
            "groundtruth": groundtruth,
            "answer": answer_text,
            "question_id": question_id
        }
        all_answers.append(question_dict)
    
    return all_answers

# Log into Huggingface 
token = "" 
login(token = token)

torch.cuda.empty_cache()
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

parser = argparse.ArgumentParser("")
# parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                    help='Enter the full model name or path')
parser.add_argument("--download_dir", type=str, default='/shared/3/cache/huggingface', 
                    help='Enter the dir for model cache')
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument('--dataset_lst', type=str, 
                    help='Comma-separated list of datasets')
parser.add_argument('--role_prompt', type=int, default=1, 
                    help='Set role_prompt to 1 to use persona in system prompts. Set to 0 to disable personas in system prompts.') 

'''LOAD ARGS'''
args = parser.parse_args()
print(args)

# Get the number of GPUs available from CUDA_VISIBLE_DEVICES
cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', "")
if cuda_visible_devices:
    args.tensor_parallel_size = len(cuda_visible_devices.split(","))
else:
    args.tensor_parallel_size = 1

# use_cuda = args.use_cuda
tensor_parallel_size = args.tensor_parallel_size
download_dir = args.download_dir
model_name = args.model_name
role_prompt = args.role_prompt

'''LOAD QUESTIONS'''
dataset_path = os.path.join(parent_dir, 'data/final_mmlu_sample_ques.csv')
question_df = pd.read_csv(dataset_path)
if args.dataset_lst:
    dataset_lst = [item.strip() for item in args.dataset_lst.split(',')]
else:
    dataset_lst = list(set(pd.read_csv(dataset_path)['dataset'].tolist()))

'''ROLES & PROMPTS'''
### load lists of prompts and roles 
if role_prompt == 1:
    prompts = ["Imagine you are talking to {indefinite} {role}.",
              "You are talking to {indefinite} {role}.",
              "You are {indefinite} {role}.",
              "Imagine you are {indefinite} {role}."]
else:
    prompts = None 

role_path = os.path.join(parent_dir, 'data/final-final-role-cate.csv')
role_df = pd.read_csv(role_path)

### persona
roles = role_df['role'].unique().tolist()

model_name_dic = {'flan-t5': ["google/flan-t5-xxl"], 
#                   'llama2': ['/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'],
                  'llama3': ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"], 
                  # 'llama70': ["meta-llama/Meta-Llama-3-70B-Instruct"],
                  'opt': ["facebook/opt-iml-max-1.3b"],
                  'mistral': ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.1"],
                  'qwen': ["Qwen/Qwen1.5-7B-Chat"],
                  'qwen2.5':["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 
                             "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
                 }

'''CHOOSE MODEL'''
model = get_key_for_value(model_name_dic, model_name)

print("Model name is", model_name)
print("Model type is", model)

'''INITIALIZE vLLM OBJECT'''
generator = init_pipeline(model_name, model_name_dic, tensor_parallel_size, download_dir)
sampling_params = SamplingParams(n=1, repetition_penalty=1.1, max_tokens=32)

for dataset in dataset_lst:
    print(f"-----Processing dataset {dataset}-----")
    data = question_df[question_df['dataset'] == dataset]
    
    all_answers = []
    if prompts: 
        for role in tqdm(roles):
            indefinite = get_indefinite_article(role)
            for prompt in prompts:
                prompts_lst = process_prompts(data, model, prompt, role, indefinite)
                answers = generate_answers(data, prompts_lst, model, model_name_dic, generator, sampling_params)
                for ans in answers:
                    ans.update({"role": role, "prompt": prompt})
                all_answers.extend(answers)
    else:
        prompts_lst = process_prompts(data, model) 
        answers = generate_answers(data, prompts_lst, model, model_name_dic, generator, sampling_params)
        for ans in answers:
            ans.update({"role": "no role", "prompt": "no role"})
        all_answers.extend(answers)
        
    answers_df = pd.DataFrame(all_answers)
    if role_prompt:
        file_name = f"data/{model_name}_results/{model}_{dataset}.json"
    else: 
        file_name = f"data/{model_name}_results/no_role/{model}_{dataset}.json"
        
    output_path = os.path.join(parent_dir, file_name)
    
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    answers_df.to_json(output_path, orient='records', lines=True)
    











