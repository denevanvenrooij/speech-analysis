from paths import *

exercises = {'MPT','SEN','SPN','VOW'}
patient_ids = {'1234567','1234568','1234569','1234570','1234571'}

def new_patient(patient_ids):
    for patient_id in patient_ids:
        for exercise in exercises:
            audio_path = audio_dir / exercise / patient_id
            audio_path.mkdir(exist_ok=True, parents=True)
            processed_path = processed_dir / exercise / patient_id
            processed_path.mkdir(exist_ok=True, parents=True)
            segments_path = segments_dir / exercise / patient_id
            segments_path.mkdir(exist_ok=True, parents=True)
            # best_segments_path = best_segments_dir / exercise / patient_id
            # best_segments_path.mkdir(exist_ok=True, parents=True)
            features_path = features_dir / exercise / patient_id
            features_path.mkdir(exist_ok=True, parents=True)
            embeddings_path = embeddings_dir / exercise / patient_id
            embeddings_path.mkdir(exist_ok=True, parents=True)
            
    print(f'Created folders for {patient_ids}')
            
if __name__ == '__main__':
    new_patient(patient_ids=patient_ids)
    
    df_features_dir.mkdir(exist_ok=True, parents=True)
    df_embeddings_dir.mkdir(exist_ok=True, parents=True)  
    
    models_dir.mkdir(exist_ok=True, parents=True)
    predictions_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)