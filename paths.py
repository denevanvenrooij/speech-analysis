from pathlib import Path

audio_dir = Path('audio_files_original/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')
features_dir = Path('extracted_features/')
plots_dir = Path('plots/')
df_features_dir = Path('dataframes_features/')
models_dir = Path('models/')
predictions_dir = Path('predictions/')

exercises = {'MPT', 'SEN', 'SPN', 'VOW'}

microphones = {'mic_1', 'mic_2'}

vowels = ['i', 'e', 'a', 'o', 'u']