import os
import torch
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

def compute_similarity(sentence1, sentence2, model):
    with torch.no_grad():
        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        embedding2 = model.encode(sentence2, convert_to_tensor=True)
        
    embedding1 = embedding1.cpu().numpy()
    embedding2 = embedding2.cpu().numpy()

    embedding1_2d = embedding1.reshape(1, -1)
    embedding2_2d = embedding2.reshape(1, -1)
    
    similarity_score = cosine_similarity(embedding1_2d, embedding2_2d)[0][0]
    return similarity_score

def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(current_dir, '..', 'data', 'sim_input_data.csv')
    test_ques = pd.read_csv(data_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    if device == "cuda":
        model = torch.nn.DataParallel(model).module

    tqdm.pandas(desc="Computing similarities")
    test_ques['prompt_ques_sim'] = test_ques.progress_apply(
        lambda row: compute_similarity(row['context_prompt'], row['full_question'], model), axis=1
    )
    
    output_file_path = os.path.join(current_dir, '..', 'data', 'testset_sim_result.csv')

    test_ques.to_csv(output_file_path, index=False)  # Replace with your desired file path

if __name__ == '__main__':
    main()