from pathlib import Path

audio_dir = Path('audio_files_original/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')

features_dir = Path('extracted_features/')
plots_dir = Path('plots/')
models_dir = Path('models/')
predictions_dir = Path('predictions/')

df_features_dir_pe = Path('dataframes_features/per_exercise/')
df_features_dir_pp = Path('dataframes_features/per_participant/')

feature_selection_dir_pe = Path('dataframes_feature_selection/per_exercise/')
feature_selection_dir_pp = Path('dataframes_feature_selection/per_participant/')


exercises = {'MPT', 'SEN', 'SPN', 'VOW'}
microphones = {'mic_1', 'mic_2'}
vowels = ['i', 'e', 'a', 'o', 'u']