from pathlib import Path
import parselmouth
from parselmouth.praat import call

audio_dir = Path('audio_files_pre/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')

vowel_dict = {
    1:'i',
    2:'e',
    3:'a',
    4:'o',
    5:'u',
}

def pre_emphasize_audio():
    original_files = [file for file in audio_dir.rglob('*') if file.is_file()]
    processed_files = [file for file in processed_dir.rglob('*') if file.is_file()]
    unprocessed_files = [file for file in original_files if file not in processed_files]
    
    for file in unprocessed_files:
        if file.suffix == ".wav":
            s = parselmouth.Sound(file)
            s.pre_emphasize()
            
            original_path = file.name
            patient_number = original_path[:7]
            exercise = original_path[10:13]
            
            output_path = Path(f'audio_files_pre/{exercise}/{patient_number}')
            if not output_path.exists():
                print(f"Path {output_path} does not exist... Did you run 0_audio_init.py?")
            
            save_path = output_path / (file.stem + "_pre.wav")
            s.save(str(save_path), 'WAV')
       
            
def save_vowels_separately(audio_file, patient_number, silence_threshold=50):
    audio_path = Path(audio_file)
    output_stem = audio_path.stem
    
    sound = parselmouth.Sound(audio_file)

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
        segment_path = segments_dir / 'VOW' / patient_number / f"{output_stem[:-1]}{label}_pre.wav"
        segment_sound.save(str(segment_path), "WAV")
        
    return segments_sorted
                 
if __name__=='__main__':
    pre_emphasize_audio()
    
    processed_files = [file for file in processed_dir.rglob('*') if file.is_file()]
    for file in processed_files:
        if 'VOW_1_pre' in file.stem:
            original_path = file.name
            patient_number = original_path[:7]
            admission_day = original_path[8:9]
            audio_path = processed_dir / 'VOW' / patient_number / f'{patient_number}_{admission_day}_VOW_1_pre.wav'
            save_vowels_separately(audio_file=str(audio_path), patient_number=patient_number, silence_threshold=50)