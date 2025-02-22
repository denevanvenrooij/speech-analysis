import os
import pandas as pd
import parselmouth
from parselmouth.praat import call

audio_files_dir = 'audio_files_original'
processed_files_dir = 'audio_files_pre'

vowel_dict = {
    1:'i',
    2:'e',
    3:'a',
    4:'o',
    5:'u',
}


def get_files(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files]
    

def pre_emphasize_audio():
    original_files = get_files(audio_files_dir)
    processed_files = get_files(processed_files_dir)
    unprocessed_files = [path for path in original_files if path not in processed_files]
    
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
       
            
def save_vowels_separately(audio_path, silence_threshold=50):
    sound = parselmouth.Sound(audio_path)

    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()    
    non_silent_mask = intensity_values > silence_threshold

    print(" ".join(f"\033[1m({time:.2f}s {int(intensity)}dB\033[0m)" if intensity >= silence_threshold 
        else f"({time:.2f}s {int(intensity)}dB)" for time, intensity in zip(time_stamps[:900], intensity_values[:900])))

    segments = []
    start = None
    silent_count = 0

    for i in range(len(non_silent_mask)):
        if non_silent_mask[i]:  
            if start is None:
                start = time_stamps[i]
            silent_count = 0
        else:  
            if start is not None:
                silent_count += 1
                if silent_count > 4: ## 5 consecutive silent timestamps marks the start of a new segment
                    end = time_stamps[i - silent_count]
                    segments.append((start, end))
                    start = None
                    silent_count = 0  

    if start is not None:  ## includes final segment when needed (if audio goes to the end of the file)
        segments.append((start, time_stamps[-1])) 

    longest_segments = sorted(segments, key=lambda x: x[1] - x[0], reverse=True)[:5] ## takes the 5 longest segments
    segments_sorted = sorted(longest_segments, key=lambda x: x[0]) ## order them again
    
    for i, (start, end) in enumerate(segments_sorted):
        print(f"Segment {i+1}: {start:.2f} {end:.2f}")

    for i, (start, end) in enumerate(segments_sorted, start=1):
        segment_sound = sound.extract_part(from_time=start, to_time=end)
        label = vowel_dict.get(i, f"{i+1}")
        output_path = audio_path.replace(".wav", "")
        output_path = f"{output_path}_{label}.wav"
        segment_sound.save(output_path, "WAV")
        
    return segments_sorted
                 
if __name__=='__main__':
    pre_emphasize_audio()
    
    processed_files = get_files(processed_files_dir)
    for file in processed_files:
        original_path = os.path.basename(file)
        patient_number = original_path[:7]
        admission_day = original_path[8:9]
        audio_path = f'{processed_files_dir}/VOW/{patient_number}/{patient_number}_{admission_day}_VOW_1_pre.wav'
        save_vowels_separately(audio_path=audio_path)