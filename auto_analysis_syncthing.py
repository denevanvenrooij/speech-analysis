from paths import *
from pathlib import Path
import parselmouth
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import re
import shutil
import subprocess

time = datetime.now().strftime('%y%m%d_%H%M%S')
log_filename = f"logs/0_aas_{time}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

folder = Path("/mnt/c/Users/denev/Sync/Syncthing")
processed_log = logs_dir / "already_processed.csv"

if processed_log.exists():
    renamed_df = pd.read_csv(processed_log)
    renamed_files = set(renamed_df["original_name"].tolist())
else:
    renamed_df = pd.DataFrame(columns=["original_name", "new_name"])
    renamed_files = set()


def extract_number(filename: str):
    match = re.search(r"\d+", filename)
    return match.group() if match else None


def rename_new_files(file_path: Path):
    original_name = file_path.name

    if original_name in renamed_files:
        return

    for name, pid in participant_dict.items():
        if name.lower() in original_name.lower():
            number = extract_number(original_name)
            if not number:
                logging.warning(f"No number found in filename: {original_name}")
                return

            new_name = f"{pid}_{number}.m4a"
            new_path = syncthing_dir / 'm4a' / new_name

            try:
                shutil.copy2(str(file_path), str(new_path))  # copy2 preserves metadata
                logging.info(f"Copied and renamed: {original_name} → {new_name}")
                renamed_df.loc[len(renamed_df)] = [original_name, new_name]
                return
            except Exception as e:
                logging.error(f"Failed to move {original_name}: {e}")
                return

    logging.info(f"No matching participant found for file: {original_name}")


def convert_m4a_to_wav(input_path):
    output_path = syncthing_dir / 'original' / (input_path.stem + ".wav")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        logging.info(f"Converted: {input_path.name} → {output_path.name}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode()
        logging.error(f"Failed to convert {input_path.name}: {error_msg}")
        return None


def pre_emphasize_audio(audio_file):
    if audio_file.suffix == ".wav":
        sound = parselmouth.Sound(str(audio_file))
        sound.pre_emphasize()
        save_path = syncthing_dir / 'pre' / (audio_file.stem + "_pre.wav")
        sound.save(str(save_path), 'WAV')   


def remove_silence_ends(audio_file, sound, silence_threshold, padding):
    logging.info(f"Removing silent edges (frames) from {audio_file}")
    
    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()

    non_silent_mask = intensity_values > silence_threshold

    try:
        first_non_silent_idx = next(i for i, val in enumerate(non_silent_mask) if val)
        last_non_silent_idx = len(non_silent_mask) - next(
            i for i, val in enumerate(reversed(non_silent_mask)) if val
        ) - 1
    except StopIteration:
        logging.info("Removing silent edges (frames), but all audio is silent...")
        return None ## when audio is silent

    start_time = max(0.0, time_stamps[first_non_silent_idx] - padding)
    end_time = min(sound.xmax, time_stamps[last_non_silent_idx] + padding)

    sound = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=False)
    
    return sound
   
    
def split_voice_exercises(audio_file, min_frames, silence_threshold, padding): 
    patient_id_take = audio_file.stem[:9]
    sound = parselmouth.Sound(str(audio_file))
    sound = remove_silence_ends(str(audio_file), sound, silence_threshold, padding)
    
    ## extracting the VOW segments
    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()
    non_silent_mask = intensity_values > silence_threshold
    
    segments = []
    start = None
    silent_count = 0
    
    for i in range(len(non_silent_mask)):
        if non_silent_mask[i]:
            if start is None:
                start = i
            silent_count = 0
        else:
            if start is not None:
                silent_count += 1
                if silent_count > 4:
                    end = i - silent_count
                    if (end - start + 1) >= min_frames: ## length of the non-silent frame
                        segments.append((start, end))
                    start = None
                    silent_count = 0

    if start is not None and (len(non_silent_mask) - start) >= min_frames: ## length of the non-silent frame 
        segments.append((start, len(non_silent_mask) - 1))

    for i, (start_idx, end_idx) in enumerate(segments[:5]):
        start_time = time_stamps[start_idx]
        end_time = time_stamps[end_idx]
        duration = end_time - start_time
        logging.info(f"Segment {i+1}: {duration:.2f}s (from {start_time:.2f}s to {end_time:.2f}s)")
    
        segment_sound = sound.extract_part(from_time=start_time, to_time=end_time)
        label = vowel_dict.get(i+1, f"{i+1}")
        segment_path = syncthing_dir / 'processed' / 'VOW' / f"{patient_id_take}_VOW_4{label}_pre.wav"
        segment_sound.save(str(segment_path), "WAV")

        new_start_time = end_time + 1.0 ## adding 1 second 
        logging.info(end_time)
        
    ## extracting the MPT segment
    sound = sound.extract_part(from_time=new_start_time, to_time=sound.xmax, preserve_times=False)    
    
    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()
    non_silent_mask = intensity_values > silence_threshold
    
    mpt_segments = []
    start = None
    silent_count = 0
    
    for i in range(len(non_silent_mask)):
        if non_silent_mask[i]:
            if start is None:
                start = i
            silent_count = 0
        else:
            if start is not None:
                silent_count += 1
                if silent_count > 4:
                    end = i - silent_count
                    if (end - start + 1) >= min_frames:
                        mpt_segments.append((start, end))
                    start = None
                    silent_count = 0
        
        if mpt_segments:
            start_idx, end_idx = mpt_segments[0]
            frame_duration = intensity.dx
            start_time = intensity.xmin + start_idx * frame_duration
            end_time = intensity.xmin + (end_idx + 1) * frame_duration
            
            mpt_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=False)
            segment_path = syncthing_dir / 'processed' / 'MPT' / f'{patient_id_take}_MPT_4_pre.wav'
            mpt_segment.save(str(segment_path), "WAV")
            
            new_start_time = end_time + 1.0
    
    ## extracting SPN (and before also SEN)
    sound = sound.extract_part(from_time=new_start_time, to_time=sound.xmax, preserve_times=False)

    sound = remove_silence_ends(str(audio_file), sound, silence_threshold=60, padding=0.1)
    
    voiced = sound.extract_part(from_time=end_time, to_time=sound.xmax, preserve_times=False)
    spn_path = syncthing_dir / 'processed' / 'SPN' / f'{patient_id_take}_SPN_4_pre.wav'
    voiced.save(str(spn_path), "WAV")
    
if __name__ == "__main__":
    logging.info("Script started.")

    # for file_path in folder.rglob("*.m4a"):
    #     logging.info(file_path)
    #     rename_new_files(file_path)
    # for file_path in (syncthing_dir / 'm4a').glob("*.m4a"):
    #     convert_m4a_to_wav(file_path)

    # renamed_df.to_csv(processed_log, index=False)
    
    processed_files = [file for file in (syncthing_dir / 'pre').glob('*_pre.wav') if file.is_file()]
    unprocessed_files = [file for file in (syncthing_dir / 'original').glob('*.wav') if file not in processed_files]
    
    for audio_file in unprocessed_files:
        logging.info(f'Processing {audio_file}')
    
        pre_emphasize_audio(audio_file)

        segments = split_voice_exercises(audio_file, min_frames=100, silence_threshold=50, padding=0.1)
    
    logging.info("Finished processing.")
