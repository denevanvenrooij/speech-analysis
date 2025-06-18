from paths import *
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import seaborn as sns

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

def feature_distance_plot(id_df, id, exercise, x_value, y_value, x_std):
    plt.figure(figsize=(12, 5))
    
    offset_map = {2: -0.25, 3: 0.25}
    color_map = {2: 'blue', 3: 'green'}
    exercise_mapped = exercise_map.get(exercise, exercise)
    
    for mic, color in color_map.items():
        mic_mapped = microphone_map.get(mic, mic)
        mic_df = id_df[id_df['mic'] == mic].copy()
        x = mic_df[x_value] + offset_map[mic]
        y = mic_df[y_value]
        plt.vlines(x, ymin=0, ymax=y, colors=color, alpha=0.35, linewidth=2.5)
        plt.plot(x, y, 'o', color=color, alpha=0.8, label=mic_mapped)

        if len(x) > 1:
            smoothed = lowess(y, x, frac=0.1, return_sorted=True)
            plt.plot(smoothed[:, 0], smoothed[:, 1], linestyle='--', color=color, linewidth=2)
    
    y_min = min(0, id_df[y_value].min())
    y_max = max(0, id_df[y_value].max())
    plt.ylim(y_min, y_max)
    
    mic1_df = id_df[id_df['mic'] == 1]
    plt.fill_between(mic1_df[x_value], mic1_df[y_value] - mic1_df[x_std], mic1_df[y_value] + mic1_df[x_std], color='gray', alpha=0.1, label='Mic 1 Variability')
    
    plt.axhline(0, color='gray', linestyle='-', linewidth=1)
    plt.title(f"Difference to the {mic_mapped[1]} (ID: {id} - Exercise: {exercise_mapped})")
    plt.xlabel("Feature Index")
    plt.ylabel("Difference Normalized Mean")
    plt.xticks(rotation=0)
    plt.xlim(left=id_df['feature_id'].min(), right=id_df['feature_id'].max())
    plt.legend(title='Mic Comparison')
    plt.tight_layout(pad=1)
    plt.grid(True, axis='y', linestyle=':', alpha=0.5)
    sns.despine()
    plt.savefig(plots_dir / f'diff_from_mic1_{id}_{exercise}.png')
    plt.show()


if __name__=="__main__":
    ## combine the dataframes
    for exercise in non_MPT_exercises:
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
        
        ## running separately for VOW and SEN/SPN
        if exercise == 'VOW':
            for vowel_type, vowel_df in combined_df.groupby('vowel'):
                ## cleaning and sorting df
                vowel_df['exercise'] = vowel_df['exercise'] + '_' + vowel_df['vowel']
                cleaned_df = drop_nan(vowel_df)
                cleaned_df = cleaned_df.drop(columns=target_columns + ['vowel'])
                df = cleaned_df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)
                
                feature_cols = df.columns[4:]
                id_cols = ['id', 'mic', 'exercise']
                
                ## normalize
                norm_df = zscore_normalize(df, feature_cols)
                
                ## create separate tables for mean and std values
                df_mean = norm_df.groupby(id_cols)[feature_cols].mean().reset_index()
                df_std = norm_df.groupby(id_cols)[feature_cols].std().reset_index()
                
                ## melt and merge both dfs together
                melted_df_mean = df_mean.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='mean')
                melted_df_std = df_std.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='std')
                melted_df = pd.merge(melted_df_mean, melted_df_std, on=['id', 'mic', 'exercise', 'feature'])
                
                ## creating tables per id/patient for plotting
                id_group = dict(tuple(melted_df.groupby('id')))
                for id, id_df in id_group.items():
                    ## calculate the difference in means for mic2/3 to mic1
                    mic1_vals = id_df[id_df['mic'] == 1][['id', 'feature', 'mean']].rename(columns={'mean': 'mic1_means'})
                    id_df = id_df.merge(mic1_vals, on=['id', 'feature'], how='left')
                    id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_means']
                    id_df = id_df.drop(columns='mic1_means')  
                    
                    ## put in the standard deviation of mic1 and sort to that
                    mic1_std = id_df[id_df['mic'] == 1].set_index(['id', 'exercise', 'feature'])['std']
                    id_df['mic1_std'] = id_df.set_index(['id', 'exercise', 'feature']).index.map(mic1_std)
                    id_df = id_df.sort_values('mic1_std', ascending=True).reset_index(drop=True)

                    ## sort the df on average mean difference to mic1 per feature for plotting
                    # avg_diff_per_feature = id_df.groupby('feature')['mean_diff_to_mic1'].mean()
                    # id_df['avg_diff_to_mic1'] = id_df['feature'].map(avg_diff_per_feature)
                    # id_df = id_df.sort_values('avg_diff_to_mic1', ascending=True).reset_index(drop=True)
                    
                    id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
                    id_df = id_df.sort_values(['feature_id','mic'], ascending=True).reset_index(drop=True)
                    print(id_df)

                    print(f"Creating a plot for {id} {exercise}")
                    feature_distance_plot(id_df, id, exercise, x_value='feature_id', y_value='mean_diff_to_mic1', x_std='mic1_std')
                                    
        # cleaned_df = drop_nan(combined_df)
        # df = cleaned_df.drop(columns=target_columns)
        # if exercise == 'VOW':
        #     df = df.sort_values(by=['id', 'mic', 'day', 'vowel']).reset_index(drop=True)
        #     df['exercise'] = df['exercise'] + '_' + df['vowel']
        #     df = df.drop(columns='vowel')
        # else:
        #     df = df.sort_values(by=['id', 'mic', 'day']).reset_index(drop=True)
        # feature_cols = df.columns[4:]
        
        # ## normalize the data before calculating means and stds
        # norm_df = zscore_normalize(df, feature_cols)
        # id_cols = ['id', 'mic', 'exercise']
        
        # ## create separate tables for mean and std values and add exercise
        # df_mean = norm_df.groupby(id_cols)[feature_cols].mean().reset_index()
        # df_std = norm_df.groupby(id_cols)[feature_cols].std().reset_index()
        
        # ## melt and merge both dfs together
        # melted_df_mean = df_mean.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='mean')
        # melted_df_std = df_std.melt(id_vars=id_cols, value_vars=feature_cols, var_name='feature', value_name='std')
        # melted_df = pd.merge(melted_df_mean, melted_df_std, on=['id', 'mic', 'exercise', 'feature'])
        # print(melted_df)
        
        # ## creating tables per id/patient for plotting
        # id_group = dict(tuple(melted_df.groupby('id')))
        # for id_val, id_df in id_group.items():
        #     ## calculate the difference in means for mic2/3 to mic1
        #     mic1_vals = id_df[id_df['mic'] == 1][['id', 'feature', 'mean']].rename(columns={'mean': 'mic1_means'})
        #     id_df = id_df.merge(mic1_vals, on=['id', 'feature'], how='left')
        #     id_df['mean_diff_to_mic1'] = id_df['mean'] - id_df['mic1_means']
        #     id_df = id_df.drop(columns='mic1_means')
        #     print(id_df)

            # avg_std_per_feature = id_df.groupby('feature')['mean_diffs_to_mic1'].mean()
            # id_df['avg_std_to_mic1'] = id_df['feature'].map(avg_std_per_feature)
        #     id_df = id_df.sort_values('avg_std_to_mic1', ascending=True).reset_index(drop=True)
        #     id_df['feature_id'] = pd.factorize(id_df['feature'])[0]
        #     print(id_df)

        #     print(f"Creating a plot for {id_val} {exercise}")
        #     feature_distance_plot(id_df, id_val, exercise)