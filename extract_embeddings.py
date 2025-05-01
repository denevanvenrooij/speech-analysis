import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import openl3
import soundfile as sf
from paths import *


def extract_embedding_openl3(dir, batch_size=32, embedding_size=512):
    files = [file for file in dir.rglob('*') if file.is_file()]
    
    for audio_file in files:
        audio, sr = sf.read(audio_file)
        emb_list, time_steps = openl3.get_audio_embedding(
            audio, sr, batch_size=batch_size, content_type='env', embedding_size=embedding_size)

        df = pd.DataFrame({'EMB': list(emb_list), 'TS': time_steps}).set_index('TS')

        df_expanded = df['EMB'].apply(pd.Series)
        df_expanded.columns = [f"EMB_{i}" for i in range(df_expanded.shape[1])]
        df = pd.concat([df_expanded, df], axis=1)
        df = df.drop(columns=['EMB'])
        
        filename = audio_file.stem.replace('_pre', '')
        filename_embedded = filename + f'_emb{embedding_size}.csv'
        
        parts = filename.split("_")
        if len(parts) != 4:
            print(f"Unexpected named audio file: {audio_file}")

        patient_id, day, exercise, take_letter = parts    
        
        save_dir = embeddings_dir / exercise / patient_id
        print(filename_embedded)

        df.to_csv(save_dir / filename_embedded, index=True)


if __name__=='__main__':
    
    directory_to_run = [
        processed_dir, 
        segments_dir, 
    ]
    
    for dir in directory_to_run:
        extract_embedding_openl3(dir, batch_size=32, embedding_size=512)
        extract_embedding_openl3(dir, batch_size=32, embedding_size=6144)