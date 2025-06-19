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
feature_extraction_dir = Path('dataframes_feature_extraction/')

pe = Path('per_exercise')
pp = Path('per_participant')

exercises = {'MPT', 'SEN', 'SPN', 'VOW'}
non_VOW_exercises = ['MPT','SEN','SPN']
non_MPT_exercises = ['SEN','SPN','VOW']
microphones = {'mic_1', 'mic_2', 'mic_3'}
vowels = ['i', 'e', 'a', 'o', 'u']
target_columns = ['target_bw', 'target_bnp']
exercise_map = {
    'MPT':'Max vowel sound',
    'VOW':'Vowel sound',
    'VOW_a':'Vowel sound A',
    'VOW_e':'Vowel sound E',
    'VOW_i':'Vowel sound I',
    'VOW_o':'Vowel sound O',
    'VOW_u':'Vowel sound U',
    'SEN':'Sentence',
    'SPN':'Spontaneous speech'}
microphone_map = {
    1:'Studio mic',
    2:'Clip-on mic',
    3:'Smartphone mic'}