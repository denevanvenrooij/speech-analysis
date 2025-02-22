import os

audio_dir = 'audio_files_original'
processed_dir = 'audio_files_pre'
exercises = {'MPT','SEN','VOW'}

def new_patient(patient_numbers):
    for patient_number in patient_numbers:
        for exercise in exercises:
            path = os.path.join(audio_dir, exercise, patient_number)
            os.makedirs(path, exist_ok=True)
            path2 = os.path.join(processed_dir, exercise, patient_number)
            os.makedirs(path2, exist_ok=True)
    print(f'Created folders for {patient_numbers}')
            
if __name__ == '__main__':
    new_patient(patient_numbers={'1234567','1234568'})