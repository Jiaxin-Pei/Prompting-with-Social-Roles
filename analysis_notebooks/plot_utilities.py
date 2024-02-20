import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys

def ensure_directory(file_name):
    """Ensure that the directory for the given file name exists. Create it if it does not."""
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name) and dir_name:
        os.makedirs(dir_name)
        
def plot_rolecate_rank(df, role_df, control_df, file_name=None):
    tdf = pd.merge(df, role_df[['role', 'interpersonal']], how='left')
    order = tdf.groupby('merged_cate')['accuracy'].mean().sort_values(ascending=False).reset_index()['merged_cate']

    grey_color = (0.5, 0.5, 0.5)
    palette = sns.color_palette("tab10")

    plt.figure(figsize=(9, 5))

    sns.barplot(y='merged_cate', x='accuracy', data=tdf, hue='interpersonal', order=order,dodge=False)
    plt.ylabel('Role Category', fontsize=15)
    plt.xlabel('Accuracy', fontsize=15)
    plt.xlim(0.29,0.41)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()  

    plt.axvline(x=control_df, color='r', linestyle='--', label='control')

    leg = plt.legend(loc='lower right', framealpha=0.5, fontsize='small')
    new_labels = ['Control Prompt', 'Occupational Role', 'Interpersonal Role']

    # Update the labels in the legend
    for text, label in zip(leg.texts, new_labels):
        text.set_text(label)
        
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')
    
    plt.show()   
    
    
def plot_acc_by_prompt(df, file_name=None):
    if file_name:
        ensure_directory(file_name)
    
    df['prompt type'] = df['prompt'].map(prompt_map)

    order = df.groupby('prompt')['accuracy'].mean().sort_values().reset_index()['prompt']

    plt.figure(figsize=(5, 3))
    sns.barplot(data=df, y='prompt', x='accuracy', order=order, hue='prompt type', dodge=False)

    plt.ylabel('Prompt', fontsize=15)
    plt.xlabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(rotation=45)
    plt.xlim(0.35, 0.39)
    plt.grid(False)

    plt.subplots_adjust(bottom=-0.17)  # Adjust as needed
    plt.legend(loc='upper center', bbox_to_anchor=(0.2, -0.18), ncol=3, fontsize=14)

    
def plot_domain_effect(df, file_name=None):   
    plt.figure(figsize=(8, 5))

    order = tdf.groupby(['domain'])['accuracy'].mean().sort_values().reset_index()['domain']

    barplot = sns.barplot(y='domain', x='accuracy', hue='in_domain', data=tdf, order=order,dodge=True)

    plt.ylabel('Domain', fontsize=16)
    plt.xlabel('Accuracy', fontsize=16)
    new_labels = ['Math', 'Natural Science', 'EECS', 'Medicine', 'Law', 'Econ', 'Psychology', 'Politics']
    plt.yticks(ticks=range(len(new_labels)), labels=new_labels)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(0.25,0.55)
    plt.grid(False)

    handles, labels = barplot.get_legend_handles_labels()
    legend = barplot.legend(handles=handles, labels=['In-Domain', 'Out-Domain'], title='Role Domain', loc='upper right')

    plt.setp(legend.get_texts(), fontsize=13)  
    plt.setp(legend.get_title(), fontsize=14)
    
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')
        
    plt.show()

    
def calculate_correlations(model_results_all):
    model_rolecate_dic = {}

    for model_name in model_results_all.keys():
        # Compute mean accuracy, sort, and rename
        model_rolecate_df = (
            model_results_all[model_name]['rolecate']['agg_mean']
            .sort_values(by='accuracy', ascending=False)
            .reset_index(drop=True)
            .rename(columns={'accuracy': f'accuracy_{model_name}'})
        )
        model_rolecate_dic[model_name] = model_rolecate_df

    # Merge dataframes and calculate ranks
    merged_df = model_rolecate_dic['llama']
    for model_name in model_rolecate_dic:
        if model_name != 'llama':
            merged_df = merged_df.merge(model_rolecate_dic[model_name], on='merged_cate', how='left')

    # Calculate ranks for all models
    for model_name in model_rolecate_dic:
        rank_col_name = f'rank_{model_name}'
        accuracy_col_name = f'accuracy_{model_name}'
        merged_df[rank_col_name] = merged_df[accuracy_col_name].rank(ascending=False, method='average')

    # Calculate correlations
    correlations = {}
    for model_name in model_rolecate_dic.keys():
        if model_name != 'llama':
            # Calculate Spearman correlation for 'llama' with other models
            spearman_corr, spearman_p_value = spearmanr(merged_df['rank_llama'], merged_df[f'rank_{model_name}'])
            correlations[('llama', model_name, 'spearman')] = (spearman_corr, spearman_p_value)

            # Calculate Pearson correlation for 'llama' with other models
            pearson_corr, pearson_p_value = pearsonr(merged_df['accuracy_llama'], merged_df[f'accuracy_{model_name}'])
            correlations[('llama', model_name, 'pearson')] = (pearson_corr, pearson_p_value)
    
    return merged_df, correlations


def plot_rolecate_corr(merged_df, corr_df, x_model, y_model, x_label, y_label, file_name=None):
    plt.figure(figsize=(8,8))

    sns.scatterplot(data=merged_df, x=f'rank_{x_model}', y=f'rank_{y_model}')

    # label each point with the corresponding 'merged_cate'
    for i, point in merged_df.iterrows():
        plt.text(point[f'rank_{x_model}'], point[f'rank_{y_model}'], str(point['merged_cate']), fontsize=14)

    # plt.title(f'Spearman Correlation of Role Category Ranks', fontsize=16)
    plt.xlabel(f'Rank in {x_label}', fontsize=15)
    plt.ylabel(f'Rank in {y_label}', fontsize=15)

    # Inverting the axis to show the highest rank (1) at the top
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    spearman_corr = corr_df[(x_model, y_model, 'spearman')][0]
    plt.annotate(f'Spearman Correlation of ranking: {spearman_corr:.2f}', xy=(0.05, 0.93), xycoords='axes fraction', 
                 fontsize=14, color="blue")

    # Annotate Pearson correlation
    pearson_corr = corr_df[(x_model, y_model, 'pearson')][0]
    plt.annotate(f'Pearson Correlation of accuracy: {pearson_corr:.2f}', xy=(0.05, 0.86), xycoords='axes fraction', 
                 fontsize=14, color="blue")
    
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()
    
def plot_hyponyms(mean_rank_per_role_model, file_name=None):
    plt.figure(figsize=(8,6))
    plot = sns.pointplot(x='Base Word', y='rank', hue='model',data=mean_rank_per_role_model)

    plt.xlabel('Base Word', fontsize=15)
    plt.ylabel('Mean Rank', fontsize=15)

    plt.xticks(rotation=45, fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()  
    plt.grid(False)

    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()

def alter_color(color, factor):
    # Convert color to HLS
    h, l, s = colorsys.rgb_to_hls(*color)
    # Modify the Lightness/Saturation
    l *= factor
    s *= factor
    # Convert back to RGB
    return colorsys.hls_to_rgb(h, l, s)

def darken_color(color, factor=0.6):
    # Convert color to HLS
    h, l, s = colorsys.rgb_to_hls(*color)
    # Reduce the Lightness
    l *= factor
    # Convert back to RGB
    return colorsys.hls_to_rgb(h, l, s)

def plot_test_acc_freq(testset_result_dic, file_name=None):
    occupation_categories = set(['econ', 'eecs', 'history', 'law', 'math', 'medicine', 'natural science', 'other occupations', 'politics', 'psychology'])

    freq_combined_df = (
        pd.concat([m for m in testset_result_dic.values()], ignore_index=True)
        .query("`N-gram Frequency (2018-2019)` != 0")
        .assign(
            cate=lambda df: df['merged_cate'].apply(lambda x: 'occupation' if x in occupation_categories else x),
            role=lambda df: df['role'].astype('category'),
            model=lambda df: df['model'].astype('category'),
            merged_cate=lambda df: df['merged_cate'].astype('category')
        )
    )
    
    markers = ['o', 's', 'D', '^', 'p', 'X', 'P', '*']

    categories = freq_combined_df['cate'].unique()
    models = ['llama', 'opt', 'flan']

    default_palette = sns.color_palette()
    num_models = len(models)
    base_colors = [default_palette[i % len(default_palette)] for i in range(num_models)]

    sns.set(style="whitegrid")

    # Initialize a matplotlib figure for combining plots
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):
        model_df = freq_combined_df[freq_combined_df['model'] == model]
        for j, category in enumerate(categories):
            # Filter the DataFrame for the current model and category
            category_df = model_df[model_df['cate'] == category]
            altered_color = alter_color(base_colors[i], 1 - j * 0.05)
            label = category if i == 0 else "_nolegend_"
            # Plot each category with a specific marker
            plt.scatter(category_df['N-gram Frequency (2018-2019)'], category_df['accuracy'], 
                        marker=markers[j % len(markers)], alpha=0.2, color=altered_color,
                        label=label)

    text_y_position = 0.15
    # Overlay regression lines for each model
    texts = []
    for i, model in enumerate(models):
        model_df = freq_combined_df[freq_combined_df['model'] == model]
        sns.regplot(x='N-gram Frequency (2018-2019)', y='accuracy', data=model_df, 
                    scatter=False, color=base_colors[i],
                    label=f'{model}')
        correlation = model_df['N-gram Frequency (2018-2019)'].corr(model_df['accuracy'])

        x_median = model_df['N-gram Frequency (2018-2019)'].median()
        y_median = np.polyval(np.polyfit(model_df['N-gram Frequency (2018-2019)'], 
                                      model_df['accuracy'], 1), x_median)

        y_position_for_annotation = y_median - (max(model_df['accuracy']) - min(model_df['accuracy'])) * 0.55

        darker_color = darken_color(base_colors[i])

        plt.text(x_median, y_position_for_annotation, f'{model}: ρ={correlation:.2f}', 
                 horizontalalignment='left', verticalalignment='bottom',
                 color=darker_color, fontsize=12)

        text_y_position -= 0.05

    legend_elements = [plt.Line2D([0], [0], marker=markers[j % len(markers)], color='w', 
                                  markerfacecolor='grey', label=category, markersize=10)
                       for j, category in enumerate(categories)]

    legend = plt.legend(handles=legend_elements, title='Role Category', 
                        loc='upper right', bbox_to_anchor=(1.01, 1.22),fontsize=13, ncol=4)

    legend.set_title(title='Role Category', prop={'size': 13})

    plt.xlabel('Log-scaled N-gram Frequency in 2018-2019', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().set_xscale('log')
    
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()  
    

def plot_test_acc_ppl(testset_result_dic, file_name=None):
    tdf = pd.concat([m['role_mean'] for m in testset_result_dic.values()])

    tdf_role = tdf.groupby(['role','model']).mean().reset_index()
    tdf_role = pd.merge(tdf_role, role_df[['role', 'merged_cate']], how='left')

    min_ppl = tdf_role.groupby('model')['ppl'].transform('min')
    max_ppl = tdf_role.groupby('model')['ppl'].transform('max')

    tdf_role['rescaled_ppl'] = (tdf_role['ppl'] - min_ppl) / (max_ppl - min_ppl)

    occupation_categories = ['econ', 'eecs', 'history', 'law', 'math', 'medicine', 'natural science', 'other occupations', 'politics', 'psychology']
    tdf_role['cate'] = tdf_role['merged_cate'].apply(lambda x: 'occupation' if x in occupation_categories else x)

    tdf_role['role'] = tdf_role['role'].astype('category')
    tdf_role['model'] = tdf_role['model'].astype('category')
    tdf_role['merged_cate'] = tdf_role['merged_cate'].astype('category')

    markers = ['o', 's', 'D', '^', 'p', 'X', 'P', '*']

    categories = tdf_role['cate'].unique()
    models = ['llama', 'opt', 'flan']

    # base_colors = sns.color_palette("hsv", len(models))
    default_palette = sns.color_palette()
    num_models = len(models)
    base_colors = [default_palette[i % len(default_palette)] for i in range(num_models)]

    sns.set(style="whitegrid")

    # Initialize a matplotlib figure for combining plots
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):
        model_df = tdf_role[tdf_role['model'] == model]
        for j, category in enumerate(categories):
            # Filter the DataFrame for the current model and category
            category_df = model_df[model_df['cate'] == category]
            altered_color = alter_color(base_colors[i], 1 - j * 0.05)
            label = category if i == 0 else "_nolegend_"
            # Plot each category with a specific marker
            plt.scatter(category_df['rescaled_ppl'], category_df['accuracy'], 
                        marker=markers[j % len(markers)], alpha=0.2, color=altered_color,
                        label=label)

    text_y_position = 0.15
    # Overlay regression lines for each model
    for i, model in enumerate(models):
        model_df = tdf_role[tdf_role['model'] == model]
        sns.regplot(x='rescaled_ppl', y='accuracy', data=model_df, 
                    scatter=False, color=base_colors[i],
                    label=f'{model}')
        correlation = model_df['rescaled_ppl'].corr(model_df['accuracy'])

        x_median = model_df['rescaled_ppl'].median()
        y_median = model_df['accuracy'].median()
        darker_color = darken_color(base_colors[i])

        y_position_for_annotation = y_median - (max(model_df['accuracy']) - min(model_df['accuracy'])) * 0.55

        plt.text(x_median, y_position_for_annotation, f'{model}: ρ={correlation:.2f}', 
                 horizontalalignment='left', verticalalignment='bottom',
                 color=darker_color, fontsize=12)

        # Update the y position for the next text
        text_y_position -= 0.05

    # plt.legend(title='Model',loc='upper right')
    legend_elements = [plt.Line2D([0], [0], marker=markers[j % len(markers)], color='w', 
                                  markerfacecolor='grey', label=category, markersize=10)
                       for j, category in enumerate(categories)]
    legend = plt.legend(handles=legend_elements, title='Role Category', loc='upper right', 
                        bbox_to_anchor=(1.01, 1.22),fontsize=13, ncol=4)

    legend.set_title(title='Role Category', prop={'size': 12})

    plt.xlabel('Rescaled Perplexity', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)

    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show() 

def plot_test_acc_sim(testset_result_dic, file_name=None):
    tdf = pd.concat([m['sim_role_mean'] for m in testset_result_dic.values()], ignore_index=True)

    occupation_categories = ['econ', 'eecs', 'history', 'law', 'math', 'medicine', 'natural science', 'other occupations', 'politics', 'psychology']
    tdf['cate'] = tdf['merged_cate'].apply(lambda x: 'occupation' if x in occupation_categories else x)

    tdf['role'] = tdf['role'].astype('category')
    tdf['model'] = tdf['model'].astype('category')
    tdf['merged_cate'] = tdf['merged_cate'].astype('category')

    markers = ['o', 's', 'D', '^', 'p', 'X', 'P', '*']

    categories = tdf['cate'].unique()
    models = ['llama', 'opt', 'flan']

    # base_colors = sns.color_palette("hsv", len(models))
    default_palette = sns.color_palette()
    num_models = len(models)
    base_colors = [default_palette[i % len(default_palette)] for i in range(num_models)]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    for i, model in enumerate(models):
        model_df = tdf[tdf['model'] == model]
        for j, category in enumerate(categories):
            category_df = model_df[model_df['cate'] == category]
            altered_color = alter_color(base_colors[i], 1 - j * 0.05)
            label = category if i == 0 else "_nolegend_"
            plt.scatter(category_df['prompt_ques_sim'], category_df['accuracy'], 
                        marker=markers[j % len(markers)], alpha=0.2, color=altered_color,
                        label=label)

    text_y_position = 0.15
    # Overlay regression lines for each model
    for i, model in enumerate(models):
        model_df = tdf[tdf['model'] == model]
        sns.regplot(x='prompt_ques_sim', y='accuracy', data=model_df, 
                    scatter=False, color=base_colors[i],
                    label=f'{model}')
        correlation = model_df['prompt_ques_sim'].corr(model_df['accuracy'])    
        x_pos = model_df['prompt_ques_sim'].median()
        y_pos = model_df['accuracy'].median()
        darker_color = darken_color(base_colors[i])

        plt.text(0.95, text_y_position, f'{model}: ρ={correlation:.2f}', 
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes, color=darker_color, fontsize=10)

        text_y_position -= 0.05

    # plt.legend(title='Model',loc='upper right')
    legend_elements = [plt.Line2D([0], [0], marker=markers[j % len(markers)], color='w', 
                                  markerfacecolor='grey', label=category, markersize=10)
                       for j, category in enumerate(categories)]
    legend = plt.legend(handles=legend_elements, title='Role Category', loc='upper right', 
                        bbox_to_anchor=(1.01, 1.22),fontsize=13, ncol=4)
    legend.set_title(title='Role Category', prop={'size': 13})

    plt.xlabel('Similarity between Context Prompts and Questions', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')

    plt.show()  
    

def plot_role_pick(df, file_name=None): 
    accuracy_columns = ['best_acc', 'best_in-domain_acc', 'most_sim_acc', 
                        'pred_best_indomain_acc', 'pred_role_acc', 'train_best_role_acc', 'random_baseline']

    # Calculate the differences
    for col in accuracy_columns:
        df[col + '-control'] = df[col] - df['control']

    # Melt the DataFrame with new difference columns
    value_vars = [col + '-control' for col in accuracy_columns]
    class_long = pd.melt(df, id_vars='full_question', 
                              value_vars=value_vars,
                              var_name='measurement', value_name='value')

    # Create the plot
    plt.figure(figsize=(8.7, 4))

    # Define the order of bars
    bar_order = [col + '-control' for col in accuracy_columns][::-1] # Reverse the order if needed
    ax = sns.barplot(y='measurement', x='value', data=class_long, order=bar_order)

    # Rename y-axis labels for readability
    new_labels = ['Best Role per Question', 'Best In-domain Role', 'Most Similar Role', 
                  'Predicted Best In-domain Role', 'Predicted Role', 
                  'Best Role on Training Set', 'Random Baseline']
    ax.set_yticklabels(new_labels[::-1]) # Reverse the labels if you reversed the bar order

    # Set labels and font sizes
    plt.ylabel('Comparison with Control', fontsize=15)
    plt.xlabel('Accuracy Difference', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    if file_name:
        ensure_directory(file_name)
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()