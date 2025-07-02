import importlib
import features
importlib.reload(features)

import numpy as np
from scipy.stats import skew, kurtosis

import parselmouth
from parselmouth.praat import call
from features import *


def PP_f0_mean(audio_file, f0_min=60, f0_max=300): ## the min/max are set based on standard for the human voice range
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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



def PP_jitter(audio_file, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    
    jitter_features = []

    for feature in jitter_feature_selection:
        if 'local' in feature:
            local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(local_jitter)
        elif 'abs' in feature:
            local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(local_absolute_jitter)
        elif 'rap' in feature:
            rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) 
            jitter_features.append(rap_jitter)
        elif 'ppq5' in feature:
            ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(ppq5_jitter)
        elif 'ddp' in feature:
            ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) 
            jitter_features.append(ddp_jitter)
            
    return jitter_features ## named 'PP_JIT'
 
def PP_jitter_murton(audio_file, segment_length=1.0, f0_min=60, f0_max=300):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
    point_process = call(middle_segment, "To PointProcess (periodic, cc)", f0_min, f0_max)

    jitter_features = []

    for feature in jitter_feature_selection:
        if 'local' in feature:
            local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(local_jitter)
        elif 'abs' in feature:
            local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(local_absolute_jitter)
        elif 'rap' in feature:
            rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) 
            jitter_features.append(rap_jitter)
        elif 'ppq5' in feature:
            ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_features.append(ppq5_jitter)
        elif 'ddp' in feature:
            ddp_jitter = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) 
            jitter_features.append(ddp_jitter)
    
    return jitter_features ## named 'PP_JIT_M'
    

def PP_lh_ratio(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
    spectrum = sound.to_spectrum()
    
    low_energy = spectrum.get_band_energy(0, 4000)
    high_energy = spectrum.get_band_energy(4000, 10000)
    
    lh_ratio = 10*np.log10(low_energy / high_energy)
    
    return lh_ratio ## named PP_LHR

def PP_LH_ratio_murton(audio_file, segment_length):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
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



def PP_CPP_mean_murton(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
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

def PP_CPP_median_murton(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
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

def PP_CPP_sd_murton(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
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



def PP_duration_with_pauses(audio_file, silence_threshold=50):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
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
    
    return duration ## named 'PP_DUR_WP' 

def PP_duration_without_pauses(audio_file, silence_threshold=50, min_silence_duration=0.5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
    intensity = sound.to_intensity()
    time_stamps = intensity.xs()
    intensity_values = intensity.values.T.flatten()
    non_silent_mask = intensity_values > silence_threshold

    speech_segments = []
    speech = False
    segment_start = None
    last_silent_time = None

    for i in range(len(non_silent_mask)):
        if non_silent_mask[i]:
            if not speech: ## merge silence when it does not exceed min_silence_duration
                if last_silent_time and (time_stamps[i] - last_silent_time) < min_silence_duration:
                    segment_start = speech_segments.pop()[0]
                else:
                    segment_start = time_stamps[i]

                speech = True
        else:
            if speech:
                segment_end = time_stamps[i - 1]
                speech_segments.append((segment_start, segment_end))
                speech = False
                last_silent_time = time_stamps[i]

    if speech and segment_start is not None: ## closes segment if sound reaches to the end of the recording
        speech_segments.append((segment_start, time_stamps[-1]))

    duration = sum(end - start for start, end in speech_segments)
    
    return duration ## named 'PP_DUR_WOP' 


    
def PP_harmonics_to_noise(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return hnr ## named 'PP_HNR'

def PP_harmonics_to_noise_murton(audio_file, segment_length):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
        
    harmonicity = call(middle_segment, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    
    return hnr ## named 'PP_HNR_M'


def PP_shimmer(audio_file, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)

    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    
    shimmer_features = []

    for feature in shimmer_feature_selection:
        if 'local' in feature:
            local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(local_shimmer)
        elif 'localdb' in feature:
            localdb_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(localdb_shimmer)
        elif 'apq3' in feature:
            apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq3_shimmer)
        elif 'apq5' in feature:
            apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq5_shimmer)
        elif 'apq11' in feature:
            apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq11_shimmer)
        elif 'dda' in feature:
            dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(dda_shimmer)
    
    return shimmer_features ## named 'PP_SHI'
    
def PP_shimmer_murton(audio_file, segment_length, f0_min, f0_max):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    duration = sound.duration
    
    middle_time = duration / 2
    half_segment = segment_length / 2
    start_time = max(0, middle_time - half_segment)
    end_time = min(duration, middle_time + half_segment)
    
    middle_segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)    
    point_process = call(middle_segment, "To PointProcess (periodic, cc)", f0_min, f0_max)
    
    shimmer_features = []

    for feature in shimmer_feature_selection:
        if 'local' in feature:
            local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(local_shimmer)
        elif 'localdb' in feature:
            localdb_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(localdb_shimmer)
        elif 'apq3' in feature:
            apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq3_shimmer)
        elif 'apq5' in feature:
            apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq5_shimmer)
        elif 'apq11' in feature:
            apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(apq11_shimmer)
        elif 'dda' in feature:
            dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_features.append(dda_shimmer)
    
    return shimmer_features ## named 'PP_SHI_M'
    


def PP_MFCC(audio_file):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    mfcc_obj = sound.to_mfcc(time_step=0.06)
    mfcc_matrix = mfcc_obj.to_matrix().values
    
    feature_values = {}
    mfc_feature_functions = {
        'mean': np.mean,
        'sd': np.std,
        'skew': skew,
        'kurt': kurtosis,
        'median': np.median,
        'min': np.min,
        'max': np.max,
    }

    for feature in mfc_feature_selection:
        if feature in mfc_feature_functions:
            feature_values[feature] = mfc_feature_functions[feature](mfcc_matrix, axis=1)

    feature_list = [feature_values[feature] for feature in mfc_feature_selection if feature in feature_values]
  
    return np.concatenate(feature_list).tolist()



def PP_glottal_formants_mean(audio_file, f0_min=60, f0_max=300, point_step=0.0025, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    formants = call(sound, "To Formant (burg)", point_step, num_formants, max_frequency, 0.025, 50)
    num_points = call(point_process, "Get number of points")
    formant_values = [[] for _ in range(num_formants)]
    
    for point in range(0,num_points):
        point += 1
        
        for j in range(1, num_formants + 1):
            t = call(point_process, "Get time from index", point)
            formant = call(formants, "Get value at time", j, t, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    mean_glottal_formants = [np.mean(f_values) if f_values else 0 for f_values in formant_values]

    return mean_glottal_formants ## named 'PP_GF_MEA'

def PP_glottal_formants_median(audio_file, f0_min=60, f0_max=300, point_step=0.0025, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    formants = call(sound, "To Formant (burg)", point_step, num_formants, max_frequency, 0.025, 50)
    num_points = call(point_process, "Get number of points")
    formant_values = [[] for _ in range(num_formants)]
    
    for point in range(0,num_points):
        point += 1
        
        for j in range(1, num_formants + 1):
            t = call(point_process, "Get time from index", point)
            formant = call(formants, "Get value at time", j, t, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    median_glottal_formants = [np.median(f_values) if f_values else 0 for f_values in formant_values]

    return median_glottal_formants ## named 'PP_GF_MEA'

def PP_glottal_formants_sd(audio_file, f0_min=60, f0_max=300, point_step=0.0025, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    formants = call(sound, "To Formant (burg)", point_step, num_formants, max_frequency, 0.025, 50)
    num_points = call(point_process, "Get number of points")
    formant_values = [[] for _ in range(num_formants)]
    
    for point in range(0,num_points):
        point += 1
        
        for j in range(1, num_formants + 1):
            t = call(point_process, "Get time from index", point)
            formant = call(formants, "Get value at time", j, t, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    sd_glottal_formants = [np.std(f_values) if f_values else 0 for f_values in formant_values]

    return sd_glottal_formants ## named 'PP_GF_SD'

def PP_formants_mean(audio_file, time_step=0.01, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    formants = call(sound, "To Formant (burg)", time_step, num_formants, max_frequency, 0.025, 50)
    
    formant_values = [[] for _ in range(num_formants)]
    num_points = int(sound.duration / time_step)
    
    for i in range(num_points):
        time = i * time_step
        
        for j in range(1, num_formants + 1):
            formant = call(formants, "Get value at time", j, time, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    mean_formants = [np.mean(f_values) if f_values else 0 for f_values in formant_values]

    return mean_formants ## named 'PP_F_MEA'

def PP_formants_median(audio_file, time_step=0.01, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    formants = call(sound, "To Formant (burg)", time_step, num_formants, max_frequency, 0.025, 50)
    
    formant_values = [[] for _ in range(num_formants)]
    num_points = int(sound.duration / time_step)
    
    for i in range(num_points):
        time = i * time_step
        
        for j in range(1, num_formants + 1):
            formant = call(formants, "Get value at time", j, time, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    median_formants = [np.median(f_values) if f_values else 0 for f_values in formant_values]

    return median_formants ## named 'PP_F_MED'

def PP_formants_sd(audio_file, time_step=0.01, max_frequency=5000, num_formants=5):
    sound = parselmouth.Sound(audio_file, sampling_frequency=48000)
    formants = call(sound, "To Formant (burg)", time_step, num_formants, max_frequency, 0.025, 50)
    
    formant_values = [[] for _ in range(num_formants)]
    num_points = int(sound.duration / time_step)
    
    for i in range(num_points):
        time = i * time_step
        
        for j in range(1, num_formants + 1):
            formant = call(formants, "Get value at time", j, time, 'Hertz', 'Linear')
            if formant and formant > 0:  
                formant_values[j-1].append(formant)

    sd_formants = [np.std(f_values) if f_values else 0 for f_values in formant_values]

    return sd_formants ## named 'PP_F_SD'