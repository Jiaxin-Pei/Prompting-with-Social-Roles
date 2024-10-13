import json
import csv
import pandas as pd
import torch
import os
import lmppl
import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_key_for_value(dic, value):
    for key, values in dic.items():
        if value in values:
            return key
    return None
       
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", 
                    help='Enter the full model name or path')


'''LOAD ARGS'''
args = parser.parse_args()
batch_size = args.batch_size
model_name = args.model_name

# change the tokenizer model

model_name_dic = {'flan-t5': ["google/flan-t5-xxl"], 
                  'llama3': ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"], 
                  'mistral': ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.1"],
                  'qwen': ["Qwen/Qwen1.5-7B-Chat"],
                  'qwen2.5':["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 
                             "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
                 }
model = get_key_for_value(model_name_dic, model_name)


scorer = lmppl.LM(model_name,
                  hf_cache_dir='/shared/3/cache/huggingface',
                  device_map="auto",
                  low_cpu_mem_usage=True)

# max length for non-law is 250, for law is 550 
# scorer_flan = EncoderDecoderLM(model_name, 
#                                max_length_encoder = 550,
#                                max_length_decoder = 550,
#                                hf_cache_dir='/shared/3/cache/huggingface',
#                                device_map="auto",
#                                low_cpu_mem_usage=True)

data_path = os.path.join(parent_dir, 'data/testset_full_prompt.csv')
ppl_test_df = pd.read_csv(data_path)

ppl_test_df['prompt_ques_ppl'] = scorer.get_perplexity(list(ppl_test_df['full_prompt']),
                                                       batch=batch_size)

# output_path = os.path.join(parent_dir, f'data/perplexity/updated/qwen2.5_7B_ppl_testset.csv')
output_path = os.path.join(parent_dir, f'data/{model}_lm_lmppl_testset.csv')
print(output_path)

directory_path = os.path.dirname(output_path)

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    
ppl_test_df.to_csv(output_path, index=False)



