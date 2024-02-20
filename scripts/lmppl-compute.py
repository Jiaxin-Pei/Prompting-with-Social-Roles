import json
import csv
import pandas as pd
import torch
import os
import lmppl
import argparse
from ppl_encoder_decoder_lm import * 

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

parser = argparse.ArgumentParser("")
parser.add_argument("--batch_size", default=4, type=int)


'''LOAD ARGS'''
args = parser.parse_args()
batch_size = args.batch_size

# change the tokenizer model
# model_name = '/shared/3/projects/mingqian/llama2_hf/llama-2-7b-chat'
model_name = "facebook/opt-iml-max-1.3b"
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

# scorer_flan = lmppl.EncoderDecoderLM(model_name, 
#                                      hf_cache_dir='/shared/3/cache/huggingface',
#                                      device_map="auto",
#                                      low_cpu_mem_usage=True)

data_path = os.path.join(parent_dir, 'data/flan_test_set_ppl_data.csv')
# all_df = pd.read_csv(data_path)
# ppl_test_df = all_df[all_df['dataset'] == 'professional_law']
ppl_test_df = pd.read_csv(data_path)

ppl_test_df['prompt_ques_ppl'] = scorer.get_perplexity(list(ppl_test_df['full_prompt']),
                                                       batch=batch_size)

output_path = os.path.join(parent_dir, 'data/opt_lm_lmppl_testset.csv')
ppl_test_df.to_csv(output_path, index=False)



