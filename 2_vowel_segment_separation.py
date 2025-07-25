from paths import *
import parselmouth
import re
import logging
from datetime import datetime

time = datetime.now().strftime('%y%m%d_%H%M%S')
log_filename = f"logs/2_vss_{time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

def save_vowels_separately(audio_file, patient_id, silence_threshold=50):
    audio_path = Path(audio_file)
    output_stem = audio_path.stem
    
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)

    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()    
    non_silent_mask = intensity_values > silence_threshold

    logging.info(" ".join(
        f"({time:.2f}s {int(intensity)}dB)" if intensity >= silence_threshold
        else f"({time:.2f}s {int(intensity)}dB)"
        for time, intensity in zip(time_stamps[:900], intensity_values[:900])))
    
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
        logging.info(f"Segment {i+1}: {start:.2f} {end:.2f}")

    for i, (start, end) in enumerate(segments_sorted, start=1):
        segment_sound = sound.extract_part(from_time=start, to_time=end)
        label = vowel_dict.get(i, f"{i+1}")
        segment_path = segments_dir / 'VOW' / patient_id / f"{output_stem[:-4]}{label}_pre.wav"
        segment_sound.save(str(segment_path), "WAV")
        
    return segments_sorted


if __name__=="__main__":
    processed_files = [file for file in processed_dir.rglob('*') if file.is_file()]
    segment_files = [file for file in segments_dir.rglob('*') if file.is_file()]

    segment_prefixes = {file.name[:15] for file in segment_files}
    unprocessed_segments = [file for file in processed_files if file.name[:15] not in segment_prefixes]
    
    for file in unprocessed_segments:
        logging.info(f"Processing vowel segments of {file}")
        if re.search(r'VOW_\d+_pre', file.stem):
            parts = file.stem.split('_')
            patient_id = parts[0]
            admission_day = parts[1]
            setting = parts[3] 
            audio_path = processed_dir / 'VOW' / patient_id / f'{patient_id}_{admission_day}_VOW_{setting}_pre.wav'
            save_vowels_separately(audio_file=str(audio_path), patient_id=patient_id, silence_threshold=50)