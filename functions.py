import parselmouth
from parselmouth.praat import call

def load_best_segment(audio_file, output_path, segment_length=1.0, step_size=0.1, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file)
    duration = sound.duration
    
    min_jitter = float('inf')
    best_segment = (0, segment_length)

    start_time = 0
    while start_time + segment_length <= duration:
        end_time = start_time + segment_length
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        
        point_process = call(segment, "To PointProcess (periodic, cc)", f0_min, f0_max)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        
        if jitter < min_jitter:
            min_jitter = jitter
            best_segment = (start_time, end_time)
        
        start_time += step_size

    best_start, best_end = best_segment
    best_segment = sound.extract_part(from_time=best_start, to_time=best_end, preserve_times=True)
    
    best_segment.save(output_path, "WAV")