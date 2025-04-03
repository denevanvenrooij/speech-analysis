import pandas as pd
import openl3
import soundfile as sf
from paths import *

def extract_embedding(audio_file, batch_size=32):
    audio, sr = sf.read(audio_file)
    print(audio_file)
    # emb_list, time_steps = openl3.get_audio_embedding(audio, sr, batch_size=batch_size)

    # df = pd.DataFrame({'EMB': list(emb_list), 'TS': time_steps}).set_index('TS')

    # df_expanded = df['EMB'].apply(pd.Series)
    # df_expanded.columns = [f"EMB_{i}" for i in range(df_expanded.shape[1])]

    # df = pd.concat([df_expanded, df], axis=1)
    # df = df.drop(columns=['EMB'])
    
    # filename = audio_file.stem.replace("_pre", "")
    # parts = filename.split("_")
    # if len(parts) != 4:
    #     print(f"Unexpected named audio file: {audio_file}")

    # patient_id, day, exercise, take_letter = parts    
    
    # storage_path = f'{}'

    # df.to_csv(embeddings_dir / )

for file in 
extract_embedding()
    
