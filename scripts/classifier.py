import os
os.environ['TRANSFORMERS_CACHE'] = '/shared/3/cache/huggingface'
import json
import csv
import pandas as pd
import random, string, re, collections
import numpy as np
import torch
from argparse import ArgumentParser

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import classification_report
from scipy import stats
from argparse import ArgumentParser
import pandas as pd
import random

from sklearn.metrics import f1_score

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)


LR = 1e-5
EPOCHS = 50
BATCH_SIZE = 64


def arguments():
    parser = ArgumentParser("")
    parser.set_defaults(show_path=False, show_similarity=False)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--predict_data_path', type=str, default=None)
    parser.add_argument('--model_type', type=str, default='multi')
    
    return parser.parse_args()

'''FOR MULTIPLECLASS ROLE PRED'''
def get_data(df, split):
    df = df[df['split']==split]
    df['accuracy'] = [eval(it) for it in df['accuracy']]
    return list(df['full_question']), list(df['accuracy'])

'''FOR DATASET PRED'''
# def get_data(df, split):
#     df = df[df['split']==split]
#     df['index'] = [it for it in df['index']]
#     return list(df['full_question']), list(df['index'])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        try:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            '''CHECK dtype - long for single class, float for multi class'''
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        except:
            print(idx)
        return item

    def __len__(self):
        return len(self.labels)
    
dataset_dict = {}

dir_path = os.path.join(parent_dir, "data/flan_role_class_agg_prompt_data.csv") #change the path to the training file

data_df = pd.read_csv(dir_path)

# data_df['split'] = split_labels(len(data_df))

print(len(data_df))
for i in ['train','val','test']:
    dataset_dict[i] = {}
    dataset_dict[i]['text'],dataset_dict[i]['labels'] = get_data(data_df, split=i)
    dataset_dict[i]['labels'] = np.array(dataset_dict[i]['labels'])
    
#     print(dataset_dict[i]['text'][:2], dataset_dict[i]['labels'][:2])

args = arguments()
save_dir = args.model_name.replace('/','-') + '-role-classifier-full-question-flan'
print('models will be saved to:', save_dir)

if args.mode == 'train':
    MODEL = args.model_name
else:
    MODEL = "./%s/best_model"%save_dir

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
train_encodings = tokenizer(dataset_dict['train']['text'], truncation=True, padding=True)
val_encodings = tokenizer(dataset_dict['val']['text'], truncation=True, padding=True)
test_encodings = tokenizer(dataset_dict['test']['text'], truncation=True, padding=True)

train_dataset = MyDataset(train_encodings, dataset_dict['train']['labels'])
val_dataset = MyDataset(val_encodings, dataset_dict['val']['labels'])
test_dataset = MyDataset(test_encodings, dataset_dict['test']['labels'])


"""## Fine-tuning

The steps above prepared the datasets in the way that the trainer is expected. Now all we need to do is create a model
to fine-tune, define the `TrainingArguments`/`TFTrainingArguments` and
instantiate a `Trainer`/`TFTrainer`.
"""

training_args = TrainingArguments(
    output_dir='./%s'%save_dir,                   # output directory
    num_train_epochs=EPOCHS,                  # total number of training epochs
    per_device_train_batch_size=BATCH_SIZE,   # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
    warmup_steps=100,                         # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                        # strength of weight decay
    logging_dir='./logs',                     # directory for storing logs
    logging_steps=100,                         # when to print log
    evaluation_strategy = "steps",
    load_best_model_at_end=True,              # load or not best model at the end
)

if args.model_type == 'multi':
    
    '''ROLE PREDICTOR'''
    num_labels = 162
    model = AutoModelForSequenceClassification.from_pretrained(MODEL,
                                                               num_labels=num_labels,
                                                               problem_type="multi_label_classification")
else:
    '''DATASET CLASSIFIER'''
    num_labels = 26
    model = AutoModelForSequenceClassification.from_pretrained(MODEL,
                                                               num_labels=num_labels,
                                                               problem_type="single_label_classification")

trainer = Trainer(
    model=model,                              # the instantiated Transformers model to be trained
    args=training_args,                       # training arguments, defined above
    train_dataset=train_dataset,              # training dataset
    eval_dataset=val_dataset                  # evaluation dataset
)


if args.mode == 'train':
    trainer.train()
    trainer.save_model("./%s/best_model"%save_dir) # save best model

                                
"""## Evaluate on Val set"""

val_preds_raw, val_labels , _ = trainer.predict(val_dataset)

if args.model_type == 'multi':
    val_preds_probs = torch.sigmoid(torch.tensor(val_preds_raw)).numpy()

    threshold = 0.5
    val_preds_binary = (val_preds_probs > threshold).astype(int)

    val_labels_np = np.array(val_labels)
    val_preds_binary_np = np.array(val_preds_binary)
    val_f1_scores = f1_score(val_labels_np, val_preds_binary_np, average='samples')
else:
    val_preds_probs = torch.softmax(torch.tensor(val_preds_raw), dim=1).numpy()

    # Use argmax to get the predicted class with the highest probability
    val_preds = np.argmax(val_preds_probs, axis=1)

    # Convert val_labels to a numpy array if it's not already
    val_labels_np = np.array(val_labels)
    val_f1_scores = f1_score(val_labels_np, val_preds, average='macro')

# texts = dataset_dict['val']['text']
# true_labels = dataset_dict['val']['labels']
# with open('./singleclass_validation_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     csvwriter = csv.writer(csvfile)
   
#     headers = ['text', 'true_labels', 'predicted_labels']
#     csvwriter.writerow(headers)
    
#     for text, true_label, pred_label in zip(texts, true_labels, val_preds_probs):
#         csvwriter.writerow([text, true_label, pred_label])
        

print("######################################")
print("Mean F1 score on Val data:", val_f1_scores)
print("######################################")

"""## Evaluate on Test set"""

test_preds_raw, test_labels , _ = trainer.predict(test_dataset)

if args.model_type == 'multi':
    test_preds_probs = torch.sigmoid(torch.tensor(test_preds_raw)).numpy()

    threshold = 0.5
    test_preds_binary = (test_preds_probs > threshold).astype(int)

    test_labels_np = np.array(test_labels)
    test_preds = np.array(test_preds_binary)

    test_f1_scores = f1_score(test_labels_np, test_preds, average='samples', zero_division=0)

    texts = dataset_dict['test']['text']
    true_labels = dataset_dict['test']['labels'] 
    
    test_pred_path = os.path.join(save_dir, f"{args.predict_data_path}.csv")
    with open(test_pred_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)

        headers = ['text', 'true_labels', 'predicted_labels']
        csvwriter.writerow(headers)

        for text, true_label, pred_label in zip(texts, true_labels, test_preds_probs):
            csvwriter.writerow([text, true_label, pred_label])
        
    
else:
    test_preds_probs = torch.softmax(torch.tensor(test_preds_raw), dim=1).numpy()
    test_preds = np.argmax(test_preds_probs, axis=1)

    test_labels_np = np.array(test_labels)
    test_f1_scores = f1_score(test_labels_np, test_preds, average='macro')  # or 'micro' or 'weighted'

    texts = dataset_dict['test']['text']
    true_labels = dataset_dict['test']['labels']

    df_test_results = pd.DataFrame({
        'text': texts,
        'true_labels': true_labels,
        'predicted_labels': test_preds  # This will be the index of the highest probability class
    })

    test_pred_path = os.path.join(save_dir, f"{args.predict_data_path}.csv")
    df_test_results.to_csv(test_pred_path, index=False, encoding='utf-8')

print("######################################")
print("Mean F1 score on Test data:", test_f1_scores)
print("######################################")





