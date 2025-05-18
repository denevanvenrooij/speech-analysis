import sys
import json
from pathlib import Path
from paths import *


patient_file = Path(__file__).parent / 'patient_ids.json'

if patient_file.exists():
    with open(patient_file, 'r') as f:
        patient_ids = set(json.load(f))
else:
    patient_ids = set()


def new_patient(patient_id):
    if patient_id in patient_ids:
        print(f"Error: Patient ID '{patient_id}' already exists.")
        return False

    for exercise in exercises:
        (audio_dir / exercise / patient_id).mkdir(exist_ok=True, parents=True)
        (processed_dir / exercise / patient_id).mkdir(exist_ok=True, parents=True)
        (segments_dir / exercise / patient_id).mkdir(exist_ok=True, parents=True)
        (features_dir / exercise / patient_id).mkdir(exist_ok=True, parents=True)
        
        for mic in microphones:
            (df_features_dir / exercise / mic / patient_id).mkdir(exist_ok=True, parents=True)
        

    patient_ids.add(patient_id)
    print(f'Created folders for patient {patient_id}')
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./0_init.py <patient_id1> <patient_id2> ...")
        sys.exit(1)

    new_ids = sys.argv[1:]
    added_any = False

    for patient_id in new_ids:
        if new_patient(patient_id):
            added_any = True

    if added_any:
        with open(patient_file, 'w') as f:
            json.dump(sorted(patient_ids), f)

    df_features_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    predictions_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)