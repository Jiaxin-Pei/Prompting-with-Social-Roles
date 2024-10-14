### When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models

This is the repository for the paper: When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models

Authors: Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, and David Jurgens

Abstract: Prompting serves as the major way humans interact with Large Language Models (LLM). Commercial AI systems commonly define the role of the LLM in system prompts. For example, ChatGPT uses ``You are a helpful assistant'' as part of its default system prompt. Despite current practices of adding personas to system prompts, it remains unclear how different personas affect a model's performance on objective tasks. In this study, we present a systematic evaluation of personas in system prompts. We curate a list of 162 roles covering 6 types of interpersonal relationships and 8 domains of expertise. Through extensive analysis of 4 popular families of LLMs and 2,410 factual questions, we demonstrate that adding personas in system prompts does not improve model performance across a range of questions compared to the control setting where no persona is added. Nevertheless, further analysis suggests that the gender, type, and domain of the persona can all influence the resulting prediction accuracies. We further experimented with a list of persona search strategies and found that, while aggregating results from the best persona for each question significantly improves prediction accuracy, automatically identifying the best persona is challenging, with predictions often performing no better than random selection. Overall, our findings suggest that while adding a persona may lead to performance gains in certain settings, the effect of each persona can be largely random. 

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
|   ├── vllm_inference_pipeline     <- Run experiments on different prompt templates and roles across various models 
|   ├── lmppl-compute              <- Get perplexity of pairs of prompt and question 
|   ├── ngram_frequency            <- Get Google ngram frequency of role word 
|   ├── similarity                 <- Compute similarity between question and prompt
|   ├── utilities                  <- Utility functions 
│
├── analysis_notebooks                   <- Jupyter notebooks for data analysis and plotting 
│   ├── classifier_training_data_process  <- Pre-process data for classifier training 
|   ├── dataset_role_preparation          <- Prepare data for experiments 
|   ├── plot                              <- Plotting codes for paper figures 
|   ├── plot_utilities                    <- Utility functions for plotting
|
└── README.md
```

<br>

### Experiment data

The experiment data is available at the shared [Google drive folder](https://drive.google.com/drive/folders/1bFXSCC-eI4V4K-JrDQREQTjgBM3ECbIj?usp=sharing). 

<br>
