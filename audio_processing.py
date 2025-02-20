import os
import pandas as pd
import parselmouth
from parselmouth.praat import call

audio_files_dir = 'audio_files_original'
processed_files_dir = 'audio_files_pre'


def get_files(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files]
    

def pre_emphasize_audio():
    original_files = get_files(audio_files_dir)
    adjusted_files = get_files(processed_files_dir)
    unprocessed_files = [path for path in original_files if path not in adjusted_files]
    
    
    for file in unprocessed_files:
        if file.endswith(".wav"):
            s = parselmouth.Sound(file)
            s.pre_emphasize()
            
            original_path = os.path.basename(file)
            patient_number = original_path[:7]
            exercise = original_path[10:13]
            
            processed_files_pt_dir = f'audio_files_pre/{exercise}/{patient_number}'
            
            save_path = os.path.join(processed_files_pt_dir, os.path.splitext(original_path)[0] + "_pre.wav")
            s.save(save_path, 'WAV')
                 
                 
if __name__=='__main__':
    pre_emphasize_audio()