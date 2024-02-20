## prepare datasets 
import json 

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_questions(all_df, dataset_name):
    df = all_df[all_df.dataset == dataset_name]
    return df['question'].tolist()

# def prepare_sciq_dataset(file_path):
#     data = load_json(file_path)
#     q_lst = [i['question'] for i in data]
#     a_lst = [i['correct_answer'] for i in data]
#     return q_lst, a_lst

# def prepare_trivia_dataset(file_path):
#     data = load_json(file_path)
#     q_lst = [i['Question'] for i in data]
#     a_lst = [i['Answer'] for i in data]
#     return q_lst, a_lst

# def prepare_web_dataset(file_path):
#     data = load_json(file_path)
#     q_lst = [i['question'] for i in data]
#     a_lst = [i['answers'][0] for i in data]
#     return q_lst, a_lst

# def prepare_wiki_dataset(file_path):
#     data = load_json(file_path)
#     q_lst = [i['question'] for i in data]
#     a_lst = [i['answer'] for i in data]
#     return q_lst, a_lst

## compute f1 scores 
def compute_f1_multiple_choice(answers, data):
    f1_scores = {}

    for relation_type in relation_dic.values():
        for relationship in relation_type:
            y_true = []
            y_pred = []

            for instance in answers:
                question = instance['question']
                for answer_dict in instance['answers']:
                    if answer_dict['relationship'] == relationship:
                        y_true.append(1)
                        # Extract the predicted answer and append to y_pred
                        y_pred.append(int(answer_dict['answer'].strip('.')))
                    
            # Compute F1 score for this relationship
            # Use 'macro' if you want to treat each class equally
            f1 = f1_score(y_true, y_pred, average='macro')  
            f1_scores[relationship] = f1
            
            f1_scores = dict(sorted(f1_scores.items(), key = lambda x:x[1], reverse=True))
    return f1_scores

def compute_f1_multi_control(results):
    predicted = [int(res['answer'].strip('.')) for res in results]
    true = [1] * len(results)
    f1 = f1_score(true, predicted, average='macro') 
    print("f1:", f1)
    return f1

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        
        ## modify the return line 
        return int(gold_toks == pred_toks), 0, 0
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def get_all_f1(groundtruth, answer):
    f1 = [compute_f1(g, a)[0] for a, g in zip(answer, groundtruth)]
    p = [compute_f1(g, a)[1] for a, g in zip(answer, groundtruth)]
    r = [compute_f1(g, a)[2] for a, g in zip(answer, groundtruth)]
    return f1, p, r

def compute_relation_f1(truth, model_answers):
    f1_results = {}
    rel_results = {}
    for i in range(len(model_answers)):
        true = truth[i]
        for ans in model_answers[i]['answers']:
            if len(ans['answer']) == 0:
                ans['answer'] = "NO ANSWER"                 
            if ans['relationship'] not in rel_results.keys():
                rel_results[ans['relationship']] = {}
                rel_results[ans['relationship']]['answers'] = [ans['answer']]
                rel_results[ans['relationship']]['groundtruth'] = [true]
            else:
                rel_results[ans['relationship']]['answers'].append(ans['answer'])
                rel_results[ans['relationship']]['groundtruth'].append(true)
    # print(rel_results.keys())
    
    for rel_lst in relation_dic.values():
        for rel in rel_lst:
            try:
                f1, p, r = get_all_f1(rel_results[rel]['groundtruth'], rel_results[rel]['answers'])
                f1_results[rel] = f1
            except TypeError:
                print("Error: Type Error.")
                print("f1 computation error:", rel)

#     f1_sorted = dict(sorted(f1_results.items(), key = lambda x:x[1], reverse=True))
    return f1_results, rel_results 

# for trivia qa 
def compute_control_f1(model_answers):
    questions = [i['question'] for i in model_answers]
    ans = [i['answer'] for i in model_answers]
    truth = []
    ## get the index of question >> get truth a_lst[q_idx]
    for i in range(len(questions)):
        q_idx = q_lst.index(questions[i])
        truth.append(a_lst[q_idx]['Value'])
    f1, p, r = get_all_f1(truth, ans)
    f1_mean = sum(f1)/len(f1)
    return f1_mean

