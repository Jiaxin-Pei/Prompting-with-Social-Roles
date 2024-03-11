### Is “A Helpful Assistant” the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts

This is the repository for the paper: Is “A Helpful Assistant” the Best Role for Large Language Models? A Systematic Evaluation of Social Roles in System Prompts

Authors: Mingqian Zheng, Jiaxin Pei and David Jurgens

Abstract: Prompting serves as the major way humans interact with Large Language Models (LLM). Commercial AI systems commonly define the role of the LLM in system prompts. For example, ChatGPT uses "You are a helpful assistant" as part of the default system prompt. But is ``a helpful assistant'' the best role for LLMs? In this study, we present a systematic evaluation of how social roles in system prompts affect model performance. We curate a list of 162 roles covering 6 types of interpersonal relationships and 8 types of occupations. Through extensive analysis of 3 popular LLMs and 2457  questions, we show that adding interpersonal roles in prompts consistently improves the models' performance over a range of questions. Moreover, while we find that using gender-neutral roles and specifying the role as the audience leads to better performances, predicting which role leads to the best performance remains a challenging task, and that frequency, similarity, and perplexity do not fully explain the effect of social roles on model performances. Our results can help inform the design of system prompts for AI systems.

The paper is available on [Arxiv](https://arxiv.org/abs/2311.10054).

### Project Structure

```                                
├── data                        <- Project data
│   ├── mmlu_sample_ques            <- Sampled MMLU questions
│   ├── question_split              <- Data split for classifier training
│   ├── role_info                   <- Role list and relevant attributes (e.g. role category, gender, frequency, etc.)
│
├── scripts                        <- Scripts to run experiments and get information
|   ├── classifier                 <- Train dataset classifier and role classifier
|   ├── llm_inference_pipeline     <- Run experiments on different prompt templates and roles across various models 
|   ├── llm_pick_role              <- Prompt LLMs to pick the best role for a given question 
|   ├── lmppl_compute              <- Get perplexity of pairs of prompt and question 
|   ├── ngram_frequency            <- Get Google ngram frequency of role word 
|   ├── ppl_encoder_decoder_lm     <- Get perplexity of encoder-decoder LLMs 
|   ├── similarity                 <- Compute similarity between question and prompt
|   ├── threads_inference          <- Run experiments on GPUs via threading 
|   ├── utilities                  <- Utility functions 
│
├── analysis_notebooks                   <- Jupyter notebooks for data analysis and plotting 
│   ├── classifier_training_data_process  <- Pre-process data for classifier training 
|   ├── dataset_role_preparation          <- Prepare data for experiments 
|   ├── gender_impact                     <- Gender impact analysis 
|   ├── role_performance_analysis         <- Analysis of role differences 
|   ├── plot_utilities                    <- Utility functions for plotting
|
└── README.md
```

<br>

### Experiment data

The experiment data is available at the shared [Google drive folder](https://drive.google.com/drive/folders/1bFXSCC-eI4V4K-JrDQREQTjgBM3ECbIj?usp=sharing). 

<br>
