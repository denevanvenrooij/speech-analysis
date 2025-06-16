from paths import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

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

def bland_altman_stats(x, y):
    mean = (x + y) / 2
    diff = x - y
    bias = np.mean(diff)
    loa = 1.96 * np.std(diff)
    return mean, diff, bias, bias - loa, bias + loa

def bland_altman_plot(mean, diff, bias, lower, upper, label, ax):
    ax.scatter(mean, diff, alpha=0.5)
    ax.axhline(bias, color='red', linestyle='--', label='Bias')
    ax.axhline(lower, color='gray', linestyle='--', label='LoA ±1.96 SD')
    ax.axhline(upper, color='gray', linestyle='--')
    ax.set_title(f'Bland-Altman: {label}')
    ax.set_xlabel('Mean of Pair')
    ax.set_ylabel('Difference of Pair')
    ax.legend()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

def zscore_normalize(df, feature_cols):
    # Normalize across all data for each feature
    return df.copy().assign(**{
        col: (df[col] - df[col].mean()) / df[col].std()
        for col in feature_cols
    })

def bland_altman_stats(x, y):
    mean = (x + y) / 2
    diff = x - y
    bias = diff.mean()
    loa = 1.96 * diff.std()
    return mean, diff, bias, bias - loa, bias + loa

def bland_altman_plot(mean, diff, bias, lower, upper, label, ax):
    ax.scatter(mean, diff, alpha=0.5, s=10)
    ax.axhline(bias, color='red', linestyle='--', label='Bias')
    ax.axhline(lower, color='gray', linestyle='--', label='LoA ±1.96 SD')
    ax.axhline(upper, color='gray', linestyle='--')
    ax.set_title(label)
    ax.set_xlabel('Mean of Pair')
    ax.set_ylabel('Difference')
    ax.legend()

def plot_all_pairs(df, feature_cols, exercise):
    mic_pairs = [(1, 2), (2, 3), (1, 3)]
    normalized_df = zscore_normalize(df, feature_cols)
    for feature in feature_cols:
        fig, axes = plt.subplots(1, len(mic_pairs), figsize=(6 * len(mic_pairs), 5))
        if len(mic_pairs) == 1:
            axes = [axes]
        for (mic1, mic2), ax in zip(mic_pairs, axes):
            x = normalized_df[normalized_df['mic'] == mic1][feature].values
            y = normalized_df[normalized_df['mic'] == mic2][feature].values
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]
            mean, diff, bias, lower, upper = bland_altman_stats(x, y)
            bland_altman_plot(mean, diff, bias, lower, upper, f'{feature}: Mic {mic1} & {mic2}', ax)
        plt.tight_layout()
        plt.savefig(plots_dir / f'bland_altman_{exercise}_{mic1}_{mic2}.png')
        plt.show()
    
    
def compute_snr_per_feature(df, feature_cols, mic_col='mic', subject_col='subject'):
    snr_dict = {}
    
    for mic in df[mic_col].unique():
        mic_df = df[df[mic_col] == mic]
        snr_dict[mic] = {}
        
        for feature in feature_cols:
            # Group by subject
            grouped = mic_df.groupby(subject_col)[feature]
            
            # Mean per subject (signal)
            subject_means = grouped.mean()
            signal_var = subject_means.var()
            
            # Variance within subject (noise)
            within_var = grouped.apply(lambda x: x.var()).mean()
            
            snr = signal_var / within_var if within_var > 0 else float('inf')
            snr_dict[mic][feature] = snr

    return pd.DataFrame(snr_dict)



if __name__=="__main__":
    for exercise in non_MPT_exercises:
        combined_dfs = []
        for mic_n in range(1, 4):
            matching_files = (df_features_dir / pe).glob(f"{exercise}_{mic_n}.csv")
            for file_path in matching_files:
                df = pd.read_csv(file_path)
                df['mic'] = mic_n
                df['exercise'] = exercise
                df = df[[df.columns[0], 'mic', 'exercise'] + [col for col in df.columns if col not in [df.columns[0], 'mic', 'exercise']]]
                combined_dfs.append(df)
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        cleaned_df = drop_nan(combined_df)

        feature_cols = cleaned_df.columns[6:]
        averaged_df = cleaned_df.groupby(['id','mic'])[feature_cols].mean().reset_index()
        
        feature_df = averaged_df.iloc[:, 2:]
        feature_cols = feature_df.columns
        
        df = zscore_normalize(feature_df, feature_cols)
        df.insert(0, 'exercise', exercise)
        df.insert(0, 'mic', averaged_df['mic'])
        df.insert(0, 'id', averaged_df['id'])
        df.reset_index(drop=True, inplace=True)
        df.set_index(['id', 'mic', 'exercise'], inplace=True)
        print(df)