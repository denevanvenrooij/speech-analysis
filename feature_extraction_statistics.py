from paths import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def zscore_normalize(df, scenario_cols):
    """Standardize values per feature using all scenario columns."""
    combined = df[scenario_cols].stack().groupby(level=0)
    means = combined.mean()
    stds = combined.std()
    return df[scenario_cols].sub(means, axis=0).div(stds, axis=0)

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

def plot_all_pairs(df, scenario_cols):
    normalized = zscore_normalize(df, scenario_cols)
    pairs = list(combinations(scenario_cols, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))

    if len(pairs) == 1:
        axes = [axes]

    for (s1, s2), ax in zip(pairs, axes):
        mean, diff, bias, lower, upper = bland_altman_stats(normalized[s1], normalized[s2])
        bland_altman_plot(mean, diff, bias, lower, upper, f'{s1} vs {s2}', ax)

    plt.tight_layout()
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
    df_path = 'test'
    df = pd.read_csv(df_path)
    plot_all_pairs(df, [microphones])
