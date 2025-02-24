from pathlib import Path

audio_dir = Path('audio_files_original/')
processed_dir = Path('audio_files_pre/')
segments_dir = Path('audio_files_segments/')
features_dir = Path('extracted_features/')

exercises = {'MPT','SEN','VOW'}

def new_patient(patient_numbers):
    for patient_number in patient_numbers:
        for exercise in exercises:
            audio_path = audio_dir / exercise / patient_number
            audio_path.mkdir(exist_ok=True, parents=True)
            processed_path = processed_dir / exercise / patient_number
            processed_path.mkdir(exist_ok=True, parents=True)
            features_path = features_dir / exercise / patient_number
            features_path.mkdir(exist_ok=True, parents=True)
            segments_path = segments_dir / exercise / patient_number
            segments_path.mkdir(exist_ok=True, parents=True)
    print(f'Created folders for {patient_numbers}')
            
if __name__ == '__main__':
    new_patient(patient_numbers={'1234567','1234568'})