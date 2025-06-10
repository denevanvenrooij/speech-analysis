from pathlib import Path

audio_dir = Path('audio_files_original/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')

features_dir = Path('extracted_features/')
plots_dir = Path('plots/')
models_dir = Path('models/')
predictions_dir = Path('predictions/')
logs_dir = Path('logs/')

df_features_dir = Path('dataframes_features/')
feature_selection_dir = Path('dataframes_feature_selection/')
pe = Path('per_exercise')
pp = Path('per_participant')


exercises = {'MPT', 'SEN', 'SPN', 'VOW'}
non_VOW_exercises = ['MPT','SEN','SPN']
non_MPT_exercises = ['SEN','SPN','VOW']
microphones = {'mic_1', 'mic_2'}
vowels = ['i', 'e', 'a', 'o', 'u']
target_columns = ['target_bw', 'target_bnp']