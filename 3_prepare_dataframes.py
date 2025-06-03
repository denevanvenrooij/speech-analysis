from paths import *
import pandas as pd
import numpy as np

non_VOW_exercises = ['MPT','SEN','SPN']
non_MPT_exercises = ['SEN','SPN','VOW']

def create_interpatient_df(mic): ## creates df with all patient data for every voice exercise separately, except VOW (pe)
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
            
            combined_df.to_csv(df_features_dir_pe / f'{exercise}_{mic}.csv', index=False)
            print(f"Saved combined dataframe for mic_{mic} and exercise {exercise}, with {len(combined_df)} rows")        
            

def create_intrapatient_df(mic): ## creates df with individual patient data from all voice exercises (pp)
    all_files = []

    for exercise in non_MPT_exercises:
        pattern = f'*_{exercise}_{mic}*.csv'
        files = list((features_dir / exercise).rglob(pattern))
        
        for file in files:
            all_files.append((exercise, file))

    patient_ids = sorted({file.name[:7] for _, file in all_files})

    for patient_id in patient_ids:
        if int(patient_id) < 1234567:
            continue  ## start at 1234567

        dfs = []
        for exercise, file in all_files:
            if file.name.startswith(patient_id):
                mic_part = file.stem.split('_')[-1]
                if mic_part.startswith(str(mic)) and len(mic_part) > len(str(mic)):
                    vowel = mic_part[len(str(mic)):]
                    exercise_label = f"{exercise}_{vowel}" ## add vowel to exercise for VOW
                else:
                    exercise_label = exercise

                df = pd.read_csv(file)
                df['exercise'] = exercise_label
                dfs.append(df)

        if not dfs:
            continue

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['id'] = patient_id

        cols = combined_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('id')))
        cols.insert(1, cols.pop(cols.index('exercise')))
        combined_df = combined_df[cols]
        
        cleaned_df = combined_df.loc[:, combined_df.isnull().sum() <= 1] ## this removes all columns with empty values for some rows
        cleaned_df.to_csv(df_features_dir_pp / f'{patient_id}_{mic}.csv', index=False)
        print(f"Saved data for mic_{mic} patient {patient_id} with {len(combined_df)} rows")


def create_interpatient_df_VOW_full(mic): ## creates df with all patient data from every VOW voice exercise (pe)
    pattern = f'*_VOW_{mic}*.csv'
    files = list((features_dir / 'VOW').rglob(pattern))

    patient_dfs = []

    for file in files:
        patient_id = file.name[:7]
        vowel = file.stem[-1]
        df = pd.read_csv(file)
        df['id'] = patient_id
        df['vowel'] = vowel
        patient_dfs.append(df)

    if patient_dfs:
        combined_df = pd.concat(patient_dfs, ignore_index=True)
    
        cols = combined_df.columns.tolist() ## putting the id column in front
        cols.insert(0, cols.pop(cols.index('id'))) 
        cols.insert(1, cols.pop(cols.index('vowel')))
        combined_df = combined_df[cols]
        
        combined_df.to_csv(df_features_dir_pe / f'VOW_{mic}.csv', index=False)
        print(f"Saved combined dataframe for mic_{mic} and all vowels, with {len(combined_df)} rows")


def create_interpatient_df_VOW(mic): ## creates df with all patient data from every VOW voice exercise for every vowel separately (pe)
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
            
            combined_df.to_csv(df_features_dir_pe / f'VOW_{mic}{vowel}.csv', index=False)
            print(f"Saved combined dataframe for mic_{mic} and vowel '{vowel}', with {len(combined_df)} rows")


def compute_random_bnp(day, admission, discharge, seed, min=200, max=8000):
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
        create_interpatient_df_VOW(mic)
        create_interpatient_df(mic)
        create_interpatient_df_VOW_full(mic)
        create_intrapatient_df(mic)