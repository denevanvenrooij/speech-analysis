import parselmouth
from parselmouth.praat import call
import numpy as np

## for extracting pitch: if the pitch is high, use max_formant = 5500, if the pitch is low use max_formant = 5000
# pitch = sound.to_pitch()
# mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
# if mean_pitch < 120:
#     max_formant = 5000
# elif mean_pitch >= 120:
#     max_formant = 5500


def PP_duration(audio_file, intensity_threshold=50): ## what should the threshold be?
    sound = parselmouth.Sound(audio_file)
    
    intensity = sound.to_intensity()
    
    time_values = intensity.xs()
    intensity_values = intensity.values.T[0]
    
    voiced_regions = time_values[intensity_values > intensity_threshold]
    
    if voiced_regions.size > 0:
        start_time = voiced_regions[0]
        end_time = voiced_regions[-1]
        duration = end_time - start_time
    else:
        duration = 0
        print("Check the audio and the threshold, the phonation time is now set to 0")
        
    return duration ## named PP_DUR


def PP_f0_mean(audio_file, f0_min=60, f0_max=300): ## the min/max are set based on standard for the human voice range
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch( ## the timestep is now 3.3ms as described by Murton (2023)
        time_step=0.005, ## this can be adjusted to increase the window
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  ## this removes unvoiced frames

    if len(f0) == 0:
        print("No frames were detected during F0 extraction, check audio file")
        return None

    mean_f0 = np.mean(f0)
    
    return mean_f0 ## named PP_F0


def PP_f0_sd(audio_file, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch( ## the timestep is now 3.3ms as described by Murton (2023)
        time_step=0.005, ## this can be adjusted to increase the window
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  ## this removes unvoiced frames

    if len(f0) == 0:
        print("No frames were detected during F0_SD extraction, check audio file")
        return None
        
    f0_sd = np.std(f0)
    
    return f0_sd ## named PP_F0_SD


def PP_f0_mean_murton(audio_file, f0_min=60, f0_max=300): ## the min/max are set based on standard for the human voice range
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch( ## the timestep is now 3.3ms as described by Murton (2023)
        time_step=0.0033, ## this can be adjusted to increase the window
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  ## this removes unvoiced frames

    if len(f0) == 0:
        print("No frames were detected during F0 extraction, check audio file")
        return None
        
    lower_bound, upper_bound = np.percentile(f0, [5, 95]) ## only includes 5-95 percentiles as described by Murton (2023)
    f0 = f0[(f0 >= lower_bound) & (f0 <= upper_bound)]

    mean_f0_murton = np.mean(f0)
    
    return mean_f0_murton ## named PP_F0_M


def PP_f0_sd_murton(audio_file, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch( ## the timestep is now 3.3ms as described by Murton (2023)
        time_step=0.0033, ## this can be adjusted to increase the window
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]  ## this removes unvoiced frames

    if len(f0) == 0:
        print("No frames were detected during F0_SD extraction, check audio file")
        return None
        
    lower_bound, upper_bound = np.percentile(f0, [5, 95]) ## only includes 5-95 percentiles as described by Murton (2023)
    f0 = f0[(f0 >= lower_bound) & (f0 <= upper_bound)]

    median_f0 = np.median(f0)
    
    f0_sd = f0[(f0 >= median_f0 - 50) & (f0 <= median_f0 + 50)] ## only values within 50 Hz from the median
    f0_sd_murton = np.std(f0_sd)
    
    return f0_sd_murton ## named PP_F0_SD_M

