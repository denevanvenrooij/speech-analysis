from paths import *
import parselmouth

def pre_emphasize_audio(file):
    if file.suffix == ".wav":
        sound = parselmouth.Sound(str(file), sampling_frequency=48000)
        sound.pre_emphasize()
        
        original_path = file.name
        patient_id = original_path[:7]
        exercise = original_path[10:13]
        
        output_path = Path(f'audio_files_pre/{exercise}/{patient_id}')
        if not output_path.exists():
            print(f"Path {output_path} does not exist... Did you run 0_init.py?")
        
        save_path = output_path / (file.stem + "_pre.wav")
        sound.save(str(save_path), 'WAV')
       
                 
if __name__=='__main__':
    original_files = [file for file in audio_dir.rglob('*') if file.is_file()]
    processed_files = [file for file in processed_dir.rglob('*') if file.is_file()]
    unprocessed_files = [file for file in original_files if file not in processed_files]
    
    for file in unprocessed_files:    
        pre_emphasize_audio(file)
        print(f"Pre-emphasized {file}")

    # print("Checking segment audio segment lengths to make sure every file is >1000ms long...")
    # segment_files_to_check = [file for file in segments_dir.rglob('*.wav') if file.is_file()]

    # for file in segment_files_to_check:
    #     sound = parselmouth.Sound(str(file), sampling_frequency=48000)
    #     duration_ms = sound.get_total_duration() * 1000

    #     if duration_ms < 2010:
    #         silence_duration = (2010 - duration_ms) / 1000.0
    #         silence = parselmouth.Sound(silence_duration, sound.sampling_frequency)
    #         padded_sound = sound.concatenate([sound, silence])
    #         padded_sound.save(str(file), 'WAV')
    #         print(f"→ Added silence to {file.name}")