from paths import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

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
        combined_df = drop_nan(combined_df)
        combined_df.to_csv(feature_extraction_dir / f'{exercise}_df.csv', index=False)
        for v in vowels:
            df_separated = combined_df[combined_df['exercise'] == f'VOW_{v}']
            df_separated.to_csv(feature_extraction_dir / f"{exercise}{v}_df.csv", index=False)
    else:
        combined_df = drop_nan(combined_df)
        combined_df = combined_df.drop(columns=target_columns)
        combined_df.to_csv(feature_extraction_dir / f'{exercise}_df.csv', index=False)

def drop_nan(feature_df):
    while feature_df.isna().any().any():
        row_nan_ratio = feature_df.isna().sum(axis=1) / feature_df.shape[1]
        col_nan_ratio = feature_df.isna().sum(axis=0) / feature_df.shape[0]
        worst_row = row_nan_ratio.idxmax()
        worst_col = col_nan_ratio.idxmax()
        if row_nan_ratio[worst_row] >= col_nan_ratio[worst_col]:
            feature_df = feature_df.drop(index=worst_row)
        else:
            feature_df = feature_df.drop(columns=worst_col)
    return feature_df

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
    plt.title(f"Difference to the Studio microphone (ID: {id}; Exercise: {exercise_mapped})")
    plt.xlabel("Feature Index")
    plt.ylabel("Difference Normalized Mean")
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    sns.despine()
    plt.savefig(plots_dir / f'diff_from_mic1_{id}_{exercise}.png')
    plt.show()


if __name__=="__main__":
    for exercise in non_MPT_exercises:
        prepare_dataframes(exercise)
    
    dfs = sorted(feature_extraction_dir.glob("*.csv"))

    for file in dfs:
        print(file)
        df = pd.read_csv(file)        
        df = df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)
        id_cols = ['id','mic','day','exercise']
        feature_cols = df.columns[4:]
    
        norm_df = zscore_normalize(df, feature_cols)
        df = norm_df.sort_values(['id','mic','day'])
                
        all_ids = df['id'].unique()
        all_mics = df['mic'].unique()
        all_exercises = df['exercise'].unique()

        full_index = pd.MultiIndex.from_product([all_ids, all_mics, all_exercises], names=['id', 'mic', 'exercise'])
        grouped = df.groupby(['id', 'mic', 'exercise'])[feature_cols].agg(['mean', 'std'])
        grouped = grouped.reindex(full_index).reset_index()

        grouped.columns = ['id', 'mic', 'exercise'] + ['_'.join(col).strip() for col in grouped.columns[3:]]

        mean_cols = [col for col in grouped.columns if col.endswith('_mean')]
        std_cols = [col for col in grouped.columns if col.endswith('_std')]
        mean_melted = grouped.melt(id_vars=['id','mic','exercise'], value_vars=mean_cols, var_name='feature', value_name='mean')
        mean_melted['feature'] = mean_melted['feature'].str.replace('_mean', '', regex=False)
        std_melted = grouped.melt(id_vars=['id','mic','exercise'], value_vars=std_cols, var_name='feature', value_name='std')
        std_melted['feature'] = std_melted['feature'].str.replace('_std', '', regex=False)

        melted_df = pd.merge(mean_melted, std_melted, on=['id','mic','exercise','feature']).fillna(0)

        for id, id_df in melted_df.groupby('id'):
            mic1_mean = id_df[id_df['mic'] == 1][['id','feature','mean']].rename(columns={'mean':'mic1_mean'})
            id_df = id_df.merge(mic1_mean, on=['id', 'feature'], how='left')
            id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_mean']
            id_df = id_df.drop(columns='mic1_mean')

            avg_std_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
            id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
            id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
            id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
            exercise = id_df['exercise'].iloc[0]

            print(f"Creating a plot for {id} {exercise}")
            feature_distance_plot(id_df, id, x_value='feature_id', y_value='mean_diff_to_mic1')
        
        # id_group = dict(tuple(melted_df.groupby('id')))
        # for id, id_df in id_group.items():
        #     ## calculate the difference in means for mic2/3 to mic1
        #     mic1_vals = id_df[id_df['mic'] == 1][['id', 'feature', 'mean']].rename(columns={'mean': 'mic1_means'})
        #     id_df = id_df.merge(mic1_vals, on=['id', 'feature'], how='left')
        #     id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_means']
        #     id_df = id_df.drop(columns='mic1_means')

        #     avg_std_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
        #     id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
        #     id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
        #     id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
        #     print(id_df)

            # ## assign feature ids and sort
            # id_df = id_df.sort_values('mic1_std', ascending=True).reset_index(drop=True)
            # id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
            # id_df = id_df.sort_values(['feature_id', 'mic'], ascending=True).reset_index(drop=True)
            # print(id_df)
                    
            # print(f"Creating a plot for {id} {exercise}")
            # feature_distance_plot(id_df, id, exercise, x_value='feature_id', y_value='mean_diff_to_mic1')
        
        # else:                            
        #     cleaned_df = drop_nan(combined_df)
        #     cleaned_df = cleaned_df.drop(columns=target_columns)
        #     df = cleaned_df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)
        #     feature_cols = df.columns[4:]
        #     id_cols = ['id', 'mic', 'exercise']
            
        #     ## normalize
        #     norm_df = zscore_normalize(df, feature_cols)
            
        #     ## create separate tables for mean and std values
        #     df_mean = norm_df.groupby(id_cols)[feature_cols].mean().reset_index()
        #     df_std = norm_df.groupby(id_cols)[feature_cols].std().reset_index()
            
        #     ## melt and merge both dfs together
        #     melted_df_mean = df_mean.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='mean')
        #     melted_df_std = df_std.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='std')
        #     melted_df = pd.merge(melted_df_mean, melted_df_std, on=['id', 'mic', 'exercise', 'feature'])
            
        #     ## creating tables per id/patient for plotting
        #     id_group = dict(tuple(melted_df.groupby('id')))
        #     for id, id_df in id_group.items():
        #         ## calculate the difference in means for mic2/3 to mic1
        #         mic1_vals = id_df[id_df['mic'] == 1][['id', 'feature', 'mean']].rename(columns={'mean': 'mic1_means'})
        #         id_df = id_df.merge(mic1_vals, on=['id', 'feature'], how='left')
        #         id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_means']
        #         id_df = id_df.drop(columns='mic1_means')  
                
        #         avg_std_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
        #         id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
        #         id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
        #         id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
        #         print(id_df)
                
        #         # ## put in the standard deviation of mic1 and sort to that
        #         # mic1_std = id_df[id_df['mic'] == 1].set_index(['id', 'exercise', 'feature'])['std']
        #         # id_df['mic1_std'] = id_df.set_index(['id', 'exercise', 'feature']).index.map(mic1_std)
        #         # id_df = id_df.sort_values('mic1_std', ascending=True).reset_index(drop=True)                    
        #         # id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
        #         # id_df = id_df.sort_values(['feature_id','mic'], ascending=True).reset_index(drop=True)
        #         # print(id_df)

        #         print(f"Creating a plot for {id} {exercise}")
        #         feature_distance_plot(id_df, id, exercise, x_value='feature_id', y_value='mean_diff_to_mic1')
                

        # # delta_dict[mic] = abs_y.reset_index(drop=True)
        # # mic1_df = id_df[id_df['mic'] == 1].copy()
        # # mic1_y = mic1_df[y_value].reset_index(drop=True)

        # # if 2 in delta_dict:
        # #     delta_to_1_mic2 = np.abs(delta_dict[2] - mic1_y)
        # #     print("Delta of Clip-on mic to Studio mic for each feature:\n", delta_to_1_mic2)
        # # if 3 in delta_dict:
        # #     delta_to_1_mic3 = np.abs(delta_dict[3] - mic1_y)
        # #     print("Delta of Smartphone mic to Studio mic for each feature:\n", delta_to_1_mic3)