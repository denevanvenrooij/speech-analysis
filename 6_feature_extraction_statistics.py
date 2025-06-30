from paths import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import pinv
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.weightstats import ttost_ind
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

pca = PCA(n_components=0.95)

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

# def drop_nan(feature_df):
#     while feature_df.isna().any().any():
#         row_nan_ratio = feature_df.isna().sum(axis=1) / feature_df.shape[1]
#         col_nan_ratio = feature_df.isna().sum(axis=0) / feature_df.shape[0]
#         worst_row = row_nan_ratio.idxmax()
#         worst_col = col_nan_ratio.idxmax()
#         if row_nan_ratio[worst_row] >= col_nan_ratio[worst_col]:
#             feature_df = feature_df.drop(index=worst_row)
#         else:
#             feature_df = feature_df.drop(columns=worst_col)
#     return feature_df

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
    

def mic_similarity_comparison_old(df, equivalence_margin=0.1):
    results = []
    
    for (exercise, id_), sub_df in df.groupby(['exercise', 'id']):
        
        if sub_df['mic'].nunique() < 2:
            print("Too little mics to compare, not running", sub_df)
            continue
        
        pivot_df = sub_df.pivot_table(index=['mic', 'day'], columns='variable', values='value')
        
        if pivot_df.shape[0] < 3:
            print("Too little samples, not running", pivot_df)
            continue
        
        pivot_df_values = pivot_df.values

        VI = pinv(np.cov(pivot_df_values.T))

        pairwise_dists = pdist(pivot_df_values, metric='mahalanobis', VI=VI)
        square_dists = squareform(pairwise_dists)

        mic_labels = pivot_df.reset_index()['mic'].values
        mic_names = np.unique(mic_labels)
        
        for mic_a, mic_b in combinations(mic_names, 2):
            a = np.where(mic_labels == mic_a)[0]
            b = np.where(mic_labels == mic_b)[0]

            between = square_dists[np.ix_(a, b)].flatten()
            within_a = square_dists[np.ix_(a, a)][np.triu_indices(len(a), k=1)]
            within_b = square_dists[np.ix_(b, b)][np.triu_indices(len(b), k=1)]

            if len(within_a) + len(within_b) < 2 or len(between) < 2:
                print("Length if two mics is less than 2, not running", pivot_df)
                continue

            within = np.concatenate([within_a, within_b])

            low, high = -equivalence_margin, equivalence_margin
            tost_stat, low_res, high_res = ttost_ind(between, within, low, high, usevar='unequal')
            pval_low = low_res[1]
            pval_high = high_res[1]
            equivalence_p = max(pval_low, pval_high)
            equivalent = equivalence_p < 0.05

            delta = np.mean(between) - np.mean(within)

            tstat, pval = ttest_ind(between, within, equal_var=False)
            similarity_score = (1 / (1 + abs(delta))) * pval ## arbitrary/non-statistical measure

            results.append({
                'id_exercise': f'{id_}_{exercise}',
                'mic_pair': f'{mic_a}-{mic_b}',
                'mean_within': np.mean(within),
                'mean_between': np.mean(between),
                'delta': delta,
                'equiv_p_value': equivalence_p,
                'equivalent': equivalent,
                'n_within': len(within),
                'n_between': len(between),
                't_statistic': tstat,
                'p_value': pval,
                'similarity_score': similarity_score,
            })

    return pd.DataFrame(results)


def mic_similarity(df):
    results = []

    if df['mic'].nunique() < 2:
        print("Too few mics to compare, skipping.")
        return pd.DataFrame()

    pivot_df = df.pivot_table(index='mic', columns='variable', values='value')

    if pivot_df.shape[0] < 2:
        print("Too few samples, skipping.")
        return pd.DataFrame()

    mic_labels = pivot_df.reset_index()['mic'].values
    mic_names = np.unique(mic_labels)

    for mic_a, mic_b in combinations(mic_names, 2):
        a = pivot_df.loc[pivot_df.index.get_level_values('mic') == mic_a].values
        b = pivot_df.loc[pivot_df.index.get_level_values('mic') == mic_b].values

        pairwise = [cosine(vec_a, vec_b) for vec_a in a for vec_b in b]
        mean_dist = np.mean(pairwise)
        mean_sim = 1 - mean_dist ## distance to similarity

        results.append({
            'mic_pair': f'{mic_a}-{mic_b}',
            'mean_cosine_distance': mean_dist,
            'mean_cosine_similarity': mean_sim,
            'n_comparisons': len(pairwise)
        })

    return pd.DataFrame(results)



if __name__=="__main__":
    for exercise in non_MPT_exercises:
        prepare_dataframes(exercise)
    
    dfs = sorted(feature_extraction_dir.glob("*.csv"))
    
    removing_file = feature_extraction_dir/ 'VOW_df.csv'
    if removing_file in dfs:
        idx = dfs.index(removing_file)
        dfs.pop(idx)

    for file in dfs:
        print(file)
        df = pd.read_csv(file)        
        df = df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)  
        id_cols = ['id','mic','day','exercise']
        feature_cols = df.columns[4:]
        
        norm_df = zscore_normalize(df, feature_cols)
        norm_df_copy = norm_df.copy()
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
            nan_features = (id_df.groupby('feature')[['mean', 'std']].apply(lambda x: x['mean'].isna().all() and x['std'].isna().all()))
            features_to_keep = nan_features[~nan_features].index
            id_df = id_df[id_df['feature'].isin(features_to_keep)]
            id_df_copy = id_df.copy()    
        
            # mic1_mean = id_df[id_df['mic'] == 1][['id','feature','mean']].rename(columns={'mean':'mic1_mean'})
            # id_df = id_df.merge(mic1_mean, on=['id', 'feature'], how='left')
            # id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_mean']
            # id_df = id_df.drop(columns='mic1_mean')

            # avg_std_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
            # id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
            # id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
            # id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
            # exercise = id_df['exercise'].iloc[0]

            # print(f"Creating a plot for {id} {exercise}")
            # feature_distance_plot(id_df, id, x_value='feature_id', y_value='mean_diff_to_mic1')
            pca_id_cols = ['id', 'mic', 'exercise', 'feature']
            id_columns = id_df_copy[pca_id_cols]
            id_df_numeric = id_df_copy.drop(columns=pca_id_cols)

            pca_result = pca.fit_transform(id_df_numeric)
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])], index=id_df_numeric.index)
            merged_df = pd.concat([id_columns, pca_df], axis=1)

            long_df = merged_df.melt(id_vars=pca_id_cols, var_name='variable', value_name='value')

            similarity_results = mic_similarity(long_df)
            print(similarity_results)