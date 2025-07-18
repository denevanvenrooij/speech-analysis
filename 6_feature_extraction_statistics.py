from paths import *
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

time = datetime.now().strftime('%y%m%d_%H%M%S')
log_filename = f"logs/6_fes_{time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

pca = PCA(n_components=0.95)

similarity_df = pd.DataFrame()

def prepare_dataframes(exercise):
    combined_dfs = []
    
    for mic_n in range(1, 4):
        matching_files = sorted((df_features_dir / pe).glob(f"{exercise}_{mic_n}.csv"))
        for file_path in matching_files:
            df = pd.read_csv(file_path)
            df['mic'] = mic_n
            df['exercise'] = exercise
            df = df[[df.columns[0], 'mic', 'exercise'] + [col for col in df.columns if col not in [df.columns[0], 'mic', 'exercise']]]
            combined_dfs.append(df)
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_df['id'] = combined_df['id'].apply(lambda x: str(x)[-2:])
    
    if exercise == 'VOW':
        combined_df['exercise'] = combined_df['exercise'] + '_' + combined_df['vowel']
        combined_df = combined_df.drop(columns=target_columns + ['vowel'])
        combined_df = fill_nan(combined_df, group_by='id') ## fills feature columns with average values per id
        combined_df.to_csv(feature_extraction_dir / f'{exercise}_df.csv', index=False)
        for v in vowels:
            df_separated = combined_df[combined_df['exercise'] == f'VOW_{v}']
            df_separated.to_csv(feature_extraction_dir / f"{exercise}{v}_df.csv", index=False)
    else:
        combined_df = fill_nan(combined_df, group_by='id') ## fills feature columns with average values per id
        combined_df = combined_df.drop(columns=target_columns)
        combined_df.to_csv(feature_extraction_dir / f'{exercise}_df.csv', index=False)

def fill_nan(df, group_by='id'):
    df_filled = df.copy()
    feature_cols = df.columns.difference(['id','mic','exercise','day'])
    df_filled[feature_cols] = df.groupby(group_by)[feature_cols].transform(lambda x: x.fillna(x.mean()))
    return df_filled

def drop_nan_cols(df, groupby, nan_threshold=0.9):
    col_nan_ratio = df.isna().mean()
    cols_to_drop = col_nan_ratio[col_nan_ratio > nan_threshold].index
    return df.drop(columns=cols_to_drop)

def zscore_normalize(df, feature_cols):
    return df.copy().assign(**{col: (df[col] - df[col].mean()) / df[col].std() for col in feature_cols})

def feature_distance_plot(id_df, id, x_value, y_value):
    plt.figure(figsize=(12, 5))

    offset_map = {2: -0.2, 3: 0.2}
    color_map = {2: 'blue', 3: 'green'}
    exercise_in_data = id_df['exercise'].iloc[0]
    exercise_mapped = exercise_map.get(exercise_in_data, exercise_in_data)

    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.axhline(1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.axhline(-1, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    
    for mic, color in color_map.items():
        mic_mapped = microphone_map.get(mic, mic)
        mic_df = id_df[id_df['mic'] == mic].copy()
        x = mic_df[x_value].reset_index(drop=True) + offset_map[mic]
        y = mic_df[y_value].reset_index(drop=True)
        abs_y = np.abs(y)

        abs_band = lowess(abs_y, x, frac=0.1, return_sorted=True)

        plt.fill_between(abs_band[:, 0], -abs_band[:, 1], abs_band[:, 1],color=color, alpha=0.25)
        plt.vlines(x, ymin=0, ymax=y, colors=color, alpha=0.3, linewidth=1.9)
        plt.plot(x, y, 'o', color=color, alpha=0.7, markersize=3, label=mic_mapped)
    
    y_lim = abs(id_df[y_value]).max() + 0.4
    plt.ylim(-y_lim, y_lim)
    plt.xlim(id_df['feature_id'].min(), id_df['feature_id'].max())
    plt.legend()
    plt.title(f"Difference to the Studio microphone (ID: {id}, Exercise: {exercise_mapped})")
    plt.xlabel("Feature Index")
    plt.ylabel("Difference Normalized Mean")
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    sns.despine()
    plt.savefig(plots_dir / f'diff_from_mic1_{id}_{exercise_in_data}.png')
    plt.show()
    

def mic_similarity(df):
    mic_pairs = [('1','2'),('1','3'),('2','3')]
    results = []
    df['mic'] = df['mic'].astype(str)

    for mic_a, mic_b in mic_pairs:
        cos_sims = []
        pivot = df.pivot(index='feature', columns='mic', values='mean')
        
        if mic_a in pivot.columns and mic_b in pivot.columns:
            vec_a = pivot[mic_a].values.reshape(1, -1)
            vec_b = pivot[mic_b].values.reshape(1, -1)

            similarity = cosine_similarity(vec_a, vec_b)[0, 0]
            cos_sims.append(similarity)

        mean_similarity = np.mean(cos_sims)
        mean_distance = 1 - mean_similarity

        results.append({
            'mic_pair': f"{mic_a}-{mic_b}",
            'mean_cosine_distance': mean_distance,
            'mean_cosine_similarity': mean_similarity})

    return pd.DataFrame(results)


def average_heatmap_dfs(dfs):
    stacked = pd.concat([df.stack().rename("value") for df in dfs], axis=0)
    averaged = stacked.groupby(level=[0, 1]).mean()
    result = averaged.unstack()
    result['avg'] = result.mean(axis=1)
    return result

def similarity_heatmap(df, exercise):
    measure_labels = {f'VOW_{v}': f'VOW {v}' for v in vowels}
    measure_labels.update({'SEN':'SEN','SPN':'SPN'})

    df_pe = df[df['exercise'] == exercise]

    heatmap_df = df_pe.pivot_table(index='mic_pair', columns='id', values='mean_cosine_distance')
    plot_df = heatmap_df.sort_index(axis=1)
    plot_df['avg'] = plot_df.mean(axis=1)
    plot_df.loc['avg'] = plot_df.mean()
    
    plt.figure(figsize=(10,3))

    ax = sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f", 
        vmin=0, vmax=2,
        cbar='viridis',
        cbar_kws={
            "label": "mean cosine distance", 
            "ticks": [0.0,0.5,1.0,1.5,2.0],
            "aspect": 20,
            "pad": 0.01,})
    
    n_rows, n_cols = plot_df.shape
    ax.axhline(n_rows - 1, color='white', linewidth=1)
    ax.axvline(n_cols - 1, color='white', linewidth=1) 
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('mean cosine distance', fontsize=9)
    colorbar.ax.tick_params(labelsize=7)
    
    plt.xticks(fontsize=7)
    plt.title(f'Mean cosine distance of {measure_labels[exercise]}', fontsize=10)
    plt.xlabel('ID', fontsize=9)
    plt.ylabel('Mic Pair', fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / f'mean_cosine_distance_heatmap_std_{exercise}.png')
    
def avg_distance_heatmap(df, processing_vow=False):
    measure = 'mean_cosine_distance'
    measure_labels = {measure: measure.replace('_', ' ')}

    dfs = []
    for exercise in df['exercise'].unique():
        df_pe = df[df['exercise'] == exercise]
        heatmap_df = df_pe.pivot(index='id', columns='mic_pair', values=measure)
        heatmap_df = heatmap_df.sort_index(axis=1)
        dfs.append(heatmap_df)

    averaged_df = average_heatmap_dfs(dfs)
    plot_df = averaged_df.T
    plot_df['avg'] = plot_df.mean(axis=1)
    
    plt.figure(figsize=(10,3))

    ax = sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f", 
        vmin=0, vmax=2,
        cmap='viridis',
        cbar_kws={"label": f"{measure_labels[measure]}", 
                    "ticks": [0.0,0.5,1.0,1.5,2.0],
                    "aspect": 20,
                    "pad": 0.01})

    n_rows, n_cols = plot_df.shape
    ax.axhline(n_rows - 1, color='white', linewidth=1)
    ax.axvline(n_cols - 1, color='white', linewidth=1) 
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(f"{measure_labels[measure]}", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)

    plt.xticks(fontsize=7)
    if processing_vow:
        plt.title(f'Average {measure_labels[measure]} between feature sets of all VOW exercises', fontsize=10)
    else:
        plt.title(f'Average {measure_labels[measure]} between feature sets of all voice exercises', fontsize=10)
    plt.xlabel('ID', fontsize=9)
    plt.ylabel('Mic Pair', fontsize=9)
    plt.tight_layout()
    if processing_vow:
        plt.savefig(plots_dir / f'{measure}_heatmap_VOW.png')
    else:
        plt.savefig(plots_dir / f'{measure}_heatmap.png') 

def std_distance_heatmap(df, processing_vow=False):
    measure = 'mean_cosine_distance'
    measure_labels = {measure: measure.replace('_', ' ')}    
    
    plot_df = (df.groupby(['id', 'mic_pair'])['mean_cosine_distance'].std().reset_index(name='std').pivot(index='mic_pair', columns='id', values='std'))
    plot_df['avg'] = plot_df.mean(axis=1)
    plot_df.loc['avg'] = plot_df.mean()
    
    plt.figure(figsize=(10,3))

    ax = sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f", 
        vmin=0, vmax=1,
        cmap='YlOrRd',
        cbar_kws={"label": f"{measure_labels[measure]}", 
                    "ticks": [0.0,0.5,1.0],
                    "aspect": 20,
                    "pad": 0.01})
    
    n_rows, n_cols = plot_df.shape
    ax.axhline(n_rows - 1, color='grey', linewidth=1)
    ax.axvline(n_cols - 1, color='grey', linewidth=1) 
    
    colorbar = ax.collections[0].colorbar
    colorbar.set_label(f"{measure_labels[measure]}", fontsize=9)
    colorbar.ax.tick_params(labelsize=7)

    plt.xticks(fontsize=7)
    if processing_vow:
        plt.title(f'Std {measure_labels[measure]} between feature sets of all VOW exercises', fontsize=10)
    else:
        plt.title(f'Std {measure_labels[measure]} between feature sets of all voice exercises', fontsize=10)
    plt.xlabel('ID', fontsize=9)
    plt.ylabel('Mic Pair', fontsize=9)
    plt.tight_layout()
    if processing_vow:
        plt.savefig(plots_dir / f'{measure}_heatmap_std_VOW.png')
    else:
        plt.savefig(plots_dir / f'{measure}_heatmap_std.png')

    
if __name__=="__main__":
    for exercise in non_MPT_exercises:
        prepare_dataframes(exercise)
    
    dfs = sorted(feature_extraction_dir.glob("*.csv"))
    
    removing_file = feature_extraction_dir/ 'VOW_df.csv'
    if removing_file in dfs:
        idx = dfs.index(removing_file)
        dfs.pop(idx)

    similarity_df = pd.DataFrame()
    vow_df = pd.DataFrame()

    for file in dfs:
        logging.info(file)
        
        df = pd.read_csv(file)        
        df = df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)  
        id_cols = ['id','mic','day','exercise']
        feature_cols = df.columns[4:]
        
        norm_df = zscore_normalize(df, feature_cols)
        norm_df = norm_df.sort_values(['id','mic','day'])
                
        all_ids = norm_df['id'].unique()
        all_mics = norm_df['mic'].unique()
        all_exercises = norm_df['exercise'].unique()

        full_index = pd.MultiIndex.from_product([all_ids, all_mics, all_exercises], names=['id', 'mic', 'exercise'])
        grouped = norm_df.groupby(['id', 'mic', 'exercise'])[feature_cols].agg(['mean', 'std'])
        grouped = grouped.reindex(full_index).reset_index()

        grouped.columns = ['id', 'mic', 'exercise'] + ['_'.join(col).strip() for col in grouped.columns[3:]]

        mean_cols = [col for col in grouped.columns if col.endswith('_mean')]
        std_cols = [col for col in grouped.columns if col.endswith('_std')]
        mean_melted = grouped.melt(id_vars=['id','mic','exercise'], value_vars=mean_cols, var_name='feature', value_name='mean')
        mean_melted['feature'] = mean_melted['feature'].str.replace('_mean', '', regex=False)
        std_melted = grouped.melt(id_vars=['id','mic','exercise'], value_vars=std_cols, var_name='feature', value_name='std')
        std_melted['feature'] = std_melted['feature'].str.replace('_std', '', regex=False)
        
        melted_df = pd.merge(mean_melted, std_melted, on=['id','mic','exercise','feature'])

        for id, id_df in melted_df.groupby('id'):
            logging.info(f"Patient {id}")
            nan_features = (id_df.groupby('feature')[['mean', 'std']].apply(lambda x: x['mean'].isna().all() and x['std'].isna().all()))
            features_to_keep = nan_features[~nan_features].index
            id_df = id_df[id_df['feature'].isin(features_to_keep)]
            cosine_df = id_df.iloc[:, :-1].copy()  
        
            # mic1_mean = id_df[id_df['mic'] == 1][['id','feature','mean']].rename(columns={'mean':'mic1_mean'})
            # id_df = id_df.merge(mic1_mean, on=['id', 'feature'], how='left')
            # id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_mean']
            # id_df = id_df.drop(columns='mic1_mean')

            # avg_std_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
            # id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
            # id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
            # id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
            # exercise = id_df['exercise'].iloc[0]

            # logging.info(f"Creating a plot for {id} {exercise} if feature_distance_plot() is unmuted")
            
            ## unmute this part for new plots
            # feature_distance_plot(id_df, id, x_value='feature_id', y_value='mean_diff_to_mic1')

            similarity_results = mic_similarity(cosine_df)
            logging.info(similarity_results)
            
            exercise = cosine_df['exercise'].unique()
            exercise = exercise[0]
            similarity_results['id'] = id
            similarity_results['exercise'] = exercise
            similarity_df = pd.concat([similarity_df, similarity_results], ignore_index=True)
        
            if exercise.startswith('VOW'):
                vow_df = pd.concat([vow_df, similarity_results], ignore_index=True)
            
        # avg_distance_heatmap(similarity_df)
        # std_distance_heatmap(similarity_df)
    
        for exercise in similarity_df['exercise'].unique():
            similarity_heatmap(similarity_df, exercise)
    
    # avg_distance_heatmap(vow_df, processing_vow=True)
    # std_distance_heatmap(vow_df, processing_vow=True)
    # vow_df.to_csv(plots_dir / 'cosine_data_vowels.csv')
    
    
    
    
    
