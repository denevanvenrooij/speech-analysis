from paths import *
import pandas as pd
import numpy as np

non_VOW_exercises = ['MPT','SEN','SPN']

def create_df_VOW(mic):    
    for vowel in vowels:
        pattern = f'*_VOW_{mic}{vowel}.csv'
        files = list((features_dir / 'VOW').rglob(pattern))

        patient_dfs = []

        for file in files:
            patient_id = file.name[:7]
            df = pd.read_csv(file)
            df['id'] = patient_id
            patient_dfs.append(df)

        if patient_dfs:
            combined_df = pd.concat(patient_dfs, ignore_index=True)
            
            cols = combined_df.columns.tolist() ## putting the id column in front
            cols.insert(0, cols.pop(cols.index('id'))) 
            combined_df = combined_df[cols]
            
            combined_df.to_csv(df_features_dir / 'VOW' / f'all_VOW_{mic}{vowel}.csv', index=False)
            print(f"Saved combined dataframe for mic_{mic} and vowel '{vowel}', with {len(combined_df)} rows")


def create_df(mic):
    for exercise in non_VOW_exercises:
        pattern = f'*_{exercise}_{mic}.csv'
        files = list((features_dir / exercise).rglob(pattern))

        patient_dfs = []
        
        for file in files:
            patient_id = file.name[:7]
            df = pd.read_csv(file)
            df['id'] = patient_id
            patient_dfs.append(df) 
                   
        if patient_dfs:
            combined_df = pd.concat(patient_dfs, ignore_index=True)
            
            cols = combined_df.columns.tolist() ## putting the id column in front
            cols.insert(0, cols.pop(cols.index('id'))) 
            combined_df = combined_df[cols]   
            
            combined_df.to_csv(df_features_dir / exercise / f'all_{exercise}_{mic}.csv', index=False)
            print(f"Saved combined dataframe for mic_{mic} and exercise {exercise}, with {len(combined_df)} rows")             
        

def compute_random_bnp(day, admission, discharge, seed, min=1000, max=8000):
    np.random.seed(seed)
    scale_range = discharge - admission or 1
    norm = (day - admission) / scale_range
    norm = np.clip(norm, 0, 1)
    upper = max - norm * (max - min) * 0.5
    lower = min + (1 - norm) * (max - min) * 0.5
    return round(np.random.uniform(lower, upper))


def compute_random_bw(day, admission, discharge, base_weight, seed, drop_min=2, drop_max=6):
    np.random.seed(seed)
    scale_range = discharge - admission or 1
    norm = (day - admission) / scale_range
    drop = np.random.uniform(drop_min, drop_max) * norm
    return round(base_weight - drop, 1)


def add_random_target_data(df_path):
    df = pd.read_csv(df_path)
    
    patient_id = df_path.parts[-2]
    seed = sum(ord(c) for c in patient_id)
    np.random.seed(seed) ## creates the same target data for each patient
    
    start = df['day'].min()
    end = df['day'].max()
    
    target_bnp_col = 'target_bnp'
    target_bw_col = 'target_bw'
    
    base_weight = np.random.uniform(60, 90)
    df[target_bw_col] = df['day'].apply(lambda d: compute_random_bw(d, start, end, base_weight, seed))
    df[target_bnp_col] = df['day'].apply(lambda c: compute_random_bnp(c, start, end, seed))
    
    cols = df.columns.tolist()
    cols.remove(target_bnp_col)
    cols.remove(target_bw_col)
    cols.insert(1, target_bnp_col)
    cols.insert(1, target_bw_col)
    df = df[cols]
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    
    df.to_csv(df_path, index=False)
    print(f"Added randomized target data to {df_path}")


if __name__ == '__main__':    
    df_list = [file for file in features_dir.rglob('*') if file.suffix == '.csv']

    for df_path in df_list:
        add_random_target_data(df_path)

    mic_numbers = [mic_num.split('_')[1] for mic_num in microphones]
    for mic in mic_numbers:
        create_df_VOW(mic)
        create_df(mic)