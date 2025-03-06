import parselmouth
from parselmouth.praat import call
import numpy as np


def PP_f0_mean(audio_file, f0_min=60, f0_max=300): ## the min/max are set based on standard for the human voice range
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc( ## using to_pitch_cc() for extra sensitivity
        time_step=0.005, ## this can be adjusted to increase the window
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0] ## this removes unvoiced frames

    if len(f0) == 0:
        print("No frames were detected during PP_F0 extraction, check audio file")
        return None

    mean_f0 = np.mean(f0)
    
    return mean_f0 ## named PP_F0

def PP_f0_median(audio_file, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc( 
        time_step=0.005, 
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        print("No frames were detected during PP_F02 extraction, check audio file")
        return None

    median_f0 = np.median(f0)
    
    return median_f0 ## named PP_F02

def PP_f0_sd(audio_file, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc(
        time_step=0.005,
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        print("No frames were detected during PP_F0_SD extraction, check audio file")
        return None
        
    f0_sd = np.std(f0)
    
    return f0_sd ## named PP_F0_SD



def PP_f0_mean_murton(audio_file, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc( 
        time_step=0.0033, ## the timestep is now 3.3ms as described by Murton 2023/2017
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        print("No frames were detected during PP_F0_M extraction, check audio file")
        return None
        
    lower_bound, upper_bound = np.percentile(f0, [5, 95]) ## only includes 5-95 percentiles as described by Murton 2023
    f0 = f0[(f0 >= lower_bound) & (f0 <= upper_bound)]

    mean_f0_murton = np.mean(f0)
    
    return mean_f0_murton ## named PP_F0_M

def PP_f0_median_murton(audio_file, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc( 
        time_step=0.0033,
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        print("No frames were detected during PP_F02_M extraction, check audio file")
        return None
        
    lower_bound, upper_bound = np.percentile(f0, [5, 95])
    f0 = f0[(f0 >= lower_bound) & (f0 <= upper_bound)]

    median_f0_murton = np.median(f0)
    
    return median_f0_murton ## named PP_F02_M

def PP_f0_sd_murton(audio_file, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file)
    
    pitch = sound.to_pitch_cc(
        time_step=0.0033,
        pitch_floor=f0_min, pitch_ceiling=f0_max)  

    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]

    if len(f0) == 0:
        print("No frames were detected during PP_F0_SD_M extraction, check audio file")
        return None
        
    lower_bound, upper_bound = np.percentile(f0, [5, 95])
    f0 = f0[(f0 >= lower_bound) & (f0 <= upper_bound)]

    median_f0 = np.median(f0)
    
    f0_sd = f0[(f0 >= median_f0 - 50) & (f0 <= median_f0 + 50)] ## only values within 50 Hz from the median (Murton 2023)
    f0_sd_murton = np.std(f0_sd)
    
    return f0_sd_murton ## named PP_F0_SD_M



def PP_jitter(audio_file, f0_min=60, f0_max=300, type='all'):
    sound = parselmouth.Sound(audio_file)
    
    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)  
    
    if type == 'local':
        return local_jitter 
    elif type == 'abs':
        return local_absolute_jitter
    elif type == 'rap':
        return rap_jitter
    elif type == 'ppq5':
        return ppq5_jitter
    elif type == 'ddp':
        return ddp_jitter   
    elif type == 'all':
        return local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter, ddp_jitter ## named PP_JIT 
 
def PP_jitter_murton(audio_file, segment_length=1.0, f0_min=60, f0_max=300, type='all'):
    sound = parselmouth.Sound(audio_file)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
    
    point_process = call(middle_segment, "To PointProcess (periodic, cc)", f0_min, f0_max)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)  
    
    if type == 'local':
        return local_jitter 
    elif type == 'abs':
        return local_absolute_jitter
    elif type == 'rap':
        return rap_jitter
    elif type == 'ppq5':
        return ppq5_jitter
    elif type == 'ddp':
        return ddp_jitter   
    elif type == 'all':
        return local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter, ddp_jitter ## named PP_JIT_M
   
    

def PP_lh_ratio(audio_file):
    sound = parselmouth.Sound(audio_file)
    
    spectrum = sound.to_spectrum()
    
    low_energy = spectrum.get_band_energy(0, 4000)
    high_energy = spectrum.get_band_energy(4000, 10000)
    
    lh_ratio = 10*np.log10(low_energy / high_energy)
    
    return lh_ratio ## named PP_LHR

def PP_lh_ratio_murton(audio_file, segment_length):
    sound = parselmouth.Sound(audio_file)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        
    spectrum = middle_segment.to_spectrum()
    
    low_energy = spectrum.get_band_energy(0, 4000)
    high_energy = spectrum.get_band_energy(4000, 10000)
    
    lh_ratio = 10*np.log10(low_energy / high_energy)
    
    return lh_ratio ## named PP_LHR_M



def PP_cpp_mean_murton(audio_file):
    sound = parselmouth.Sound(audio_file)
    duration = sound.get_total_duration()
    sampling_rate = sound.sampling_frequency

    cpp_values = []
    window_length = 40.96e-3 ## 40.96 ms
    step_size = 10.24e-3     ## 10.24 ms

    for start_time in np.arange(0, duration - window_length, step_size):
        frame = sound.extract_part(start_time, start_time + window_length)
        spectrum = frame.to_spectrum()
        log_spectrum = np.log(np.sum(spectrum.values**2, axis=0) + 1e-10)  ## apparently this avoid log(0)
        
        cepstrum = np.fft.irfft(log_spectrum)[:len(log_spectrum) * 2 - 2]
        quefrencies = np.arange(len(cepstrum)) / sampling_rate
        quefrency_min = 3.3e-3   ## 300 Hz
        quefrency_max = 16.7e-3  ##  60 Hz

        quefrency_indices = np.where((quefrencies >= quefrency_min) & (quefrencies <= quefrency_max))[0]
        if not len(quefrency_indices): 
            continue  ## skips empty frames

        peak_index = quefrency_indices[np.argmax(cepstrum[quefrency_indices])]
        cpp_peak = cepstrum[peak_index]

        noise_floor = np.median(cepstrum[quefrency_indices]) ## when this is not good enough use LR

        cpp_values.append(cpp_peak - noise_floor)
        mean_cpp = np.mean(cpp_values)

    return mean_cpp ## named PP_CPP_M

def PP_cpp_median_murton(audio_file):
    sound = parselmouth.Sound(audio_file)
    duration = sound.get_total_duration()
    sampling_rate = sound.sampling_frequency

    cpp_values = []
    window_length = 40.96e-3 ## 40.96 ms
    step_size = 10.24e-3     ## 10.24 ms

    for start_time in np.arange(0, duration - window_length, step_size):
        frame = sound.extract_part(start_time, start_time + window_length)
        spectrum = frame.to_spectrum()
        log_spectrum = np.log(np.sum(spectrum.values**2, axis=0) + 1e-10)  ## apparently this avoid log(0)
        
        cepstrum = np.fft.irfft(log_spectrum)[:len(log_spectrum) * 2 - 2]
        quefrencies = np.arange(len(cepstrum)) / sampling_rate
        quefrency_min = 3.3e-3   ## 300 Hz
        quefrency_max = 16.7e-3  ##  60 Hz

        quefrency_indices = np.where((quefrencies >= quefrency_min) & (quefrencies <= quefrency_max))[0]
        if not len(quefrency_indices): 
            continue  ## skips empty frames

        peak_index = quefrency_indices[np.argmax(cepstrum[quefrency_indices])]
        cpp_peak = cepstrum[peak_index]

        noise_floor = np.median(cepstrum[quefrency_indices]) ## when this is not good enough use LR

        cpp_values.append(cpp_peak - noise_floor)
        median_cpp = np.median(cpp_values)

    return median_cpp ## named PP_CPP_M2

def PP_cpp_sd_murton(audio_file):
    sound = parselmouth.Sound(audio_file)
    duration = sound.get_total_duration()
    sampling_rate = sound.sampling_frequency

    cpp_values = []
    window_length = 40.96e-3 ## 40.96 ms
    step_size = 10.24e-3     ## 10.24 ms

    for start_time in np.arange(0, duration - window_length, step_size):
        frame = sound.extract_part(start_time, start_time + window_length)
        spectrum = frame.to_spectrum()
        log_spectrum = np.log(np.sum(spectrum.values**2, axis=0) + 1e-10)  ## apparently this avoid log(0)
        
        cepstrum = np.fft.irfft(log_spectrum)[:len(log_spectrum) * 2 - 2]
        quefrencies = np.arange(len(cepstrum)) / sampling_rate
        quefrency_min = 3.3e-3   ## 300 Hz
        quefrency_max = 16.7e-3  ##  60 Hz

        quefrency_indices = np.where((quefrencies >= quefrency_min) & (quefrencies <= quefrency_max))[0]
        if not len(quefrency_indices): 
            continue  ## skips empty frames

        peak_index = quefrency_indices[np.argmax(cepstrum[quefrency_indices])]
        cpp_peak = cepstrum[peak_index]

        noise_floor = np.median(cepstrum[quefrency_indices]) ## when this is not good enough use LR

        cpp_values.append(cpp_peak - noise_floor)
        
        if not cpp_values:
            return np.nan, np.nan 
        
        std_cpp = np.std(cpp_values)

    return std_cpp ## named PP_CPP_SD_M



def PP_max_phonation(audio_file, silence_threshold=50):
    sound = parselmouth.Sound(audio_file)
    
    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()
    non_silent_mask = intensity_values > silence_threshold

    if np.any(non_silent_mask):  
        start_index = np.where(non_silent_mask)[0][0] ## first non-silent frame
        end_index = np.where(non_silent_mask)[0][-1]  ##  last non-silent frame

        start_time = time_stamps[start_index]
        end_time = time_stamps[end_index]

        duration = end_time - start_time
    else:
        duration = 0.0 ## returns this is there is no non-silence (so everything is sound)
    
    return duration ## named 'PP_MAX_PH'


    
def PP_harmonics_to_noise(audio_file):
    sound = parselmouth.Sound(audio_file)
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return hnr ## named 'PP_HNR'

def PP_harmonics_to_noise_murton(audio_file, segment_length):
    sound = parselmouth.Sound(audio_file)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        
    harmonicity = call(middle_segment, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return hnr ## named 'PP_HNR_M'


def PP_shimmer(audio_file, f0_min, f0_max, type='all'):
    sound = parselmouth.Sound(audio_file)

    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    local_shimmer =  call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdb_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer =  call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    if type == 'local':
        return local_shimmer 
    elif type == 'localdb':
        return localdb_shimmer
    elif type == 'apq3':
        return apq3_shimmer
    elif type == 'apq5':
        return aqpq5_shimmer
    elif type == 'apq11':
        return apq11_shimmer  
    elif type == 'dda':
        return dda_shimmer 
    elif type == 'all':
        return local_shimmer, localdb_shimmer, apq3_shimmer, aqpq5_shimmer, apq11_shimmer, dda_shimmer ## named PP_SHI 
    
def PP_shimmer_murton(audio_file, segment_length, f0_min, f0_max, type='all'):
    sound = parselmouth.Sound(audio_file)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)    

    point_process = call(middle_segment, "To PointProcess (periodic, cc)", f0_min, f0_max)
    local_shimmer =  call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdb_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer =  call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    if type == 'local':
        return local_shimmer 
    elif type == 'localdb':
        return localdb_shimmer
    elif type == 'apq3':
        return apq3_shimmer
    elif type == 'apq5':
        return aqpq5_shimmer
    elif type == 'apq11':
        return apq11_shimmer  
    elif type == 'dda':
        return dda_shimmer 
    elif type == 'all':
        return local_shimmer, localdb_shimmer, apq3_shimmer, aqpq5_shimmer, apq11_shimmer, dda_shimmer ## named PP_SHI_M 