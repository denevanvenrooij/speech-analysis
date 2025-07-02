from pathlib import Path

audio_dir = Path('audio_files_original/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')
syncthing_dir = Path('audio_files_sync/')

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
target_columns = ['target_bw', 'target_bnp']

vowels = ['i', 'e', 'a', 'o', 'u']
vowel_dict = {i+1: vowel for i, vowel in enumerate(vowels)}

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
    3:'Mobile phone mic'}
participant_dict = {
    "dene":"1234567",
    "puk":"1234568",
    "nico":"1234569",
    "nicolette":"1234569",
    "niels":"1234570",
    "ol":"1234571",
    "olivia":"1234571",
    "daniel":"1234572",
    "daan":"1234572",
    "juul":"1234573",
    "celestina":"1234574",
    "cel":"1234574",
    "jan":"1234575",
    "astrid":"1234576",
    "kees":"1234577",
    "angelo":"1234578"
}
