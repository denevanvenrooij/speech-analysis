AVAILABLE_FEATURES = [
    'PP_F0', 'PP_F02', 'PP_F0_SD', 
    'PP_F0_M', 'PP_F02_M', 'PP_F0_SD_M', 
    'PP_LHR', 'PP_LHR_M',
    'PP_CPP_M', 'PP_CPP_M2', 'PP_CPP_SD_M',
    'PP_DUR_WP', 'PP_DUR_WOP'
    'PP_HNR', 'PP_HNR_M',
    'PP_JIT', 'PP_JIT_M',
    'PP_SHI', 'PP_SHI_M',
    'PP_MFC',
] ## add more

selected_features_dict_VOW = {
    'PP_F0': True, 'PP_F02':True, 'PP_F0_SD': True, 
    'PP_F0_M': True, 'PP_F02_M':True, 'PP_F0_SD_M': True, 
    'PP_LHR': True, 'PP_LHR_M': True,
    'PP_CPP_M': True, 'PP_CPP_M2': True, 'PP_CPP_SD_M':True,
    'PP_DUR_WP': True, 'PP_DUR_WOP': True,
    'PP_HNR': True, 'PP_HNR_M': True,
    'PP_JIT': True, 'PP_JIT_M': True,
    'PP_SHI': True, 'PP_SHI_M': True,
} ## enable the ones you want

selected_features_dict_SEN = {
    'PP_F0': True, 'PP_F02':True, 'PP_F0_SD': True, 
    'PP_F0_M': True, 'PP_F02_M':True, 'PP_F0_SD_M': True, 
    'PP_LHR': True, 'PP_LHR_M': True,
    'PP_CPP_M': True, 'PP_CPP_M2': True, 'PP_CPP_SD_M':True,
    'PP_DUR_WP': True, 'PP_DUR_WOP': True,
    'PP_HNR': True, 'PP_HNR_M': True,
    'PP_JIT': True, 'PP_JIT_M': True,
    'PP_SHI': True, 'PP_SHI_M': True,
    'PP_MFC': True,
} ## enable the ones you want

selected_features_dict_SPN = {
    'PP_F0': True, 'PP_F02':True, 'PP_F0_SD': True, 
    'PP_F0_M': True, 'PP_F02_M':True, 'PP_F0_SD_M': True, 
    'PP_LHR': True, 'PP_LHR_M': True,
    'PP_CPP_M': True, 'PP_CPP_M2': True, 'PP_CPP_SD_M':True,
    'PP_DUR_WP': True, 'PP_DUR_WOP': True,
    'PP_HNR': True, 'PP_HNR_M': True,
    'PP_JIT': True, 'PP_JIT_M': True,
    'PP_SHI': True, 'PP_SHI_M': True,
    'PP_MFC': True,
} ## enable the ones you want

selected_features_dict_MPT = {
    'PP_DUR_WP': True, 'PP_DUR_WOP': True,
} ## enable the ones you want



jitter_feature_selection = [
    'local',
    'abs',
    'rap',
    'ppq5',
    'ddp',
] ## enable the ones you want
jitter_feature_indices = {feature: idx for idx, feature in enumerate(jitter_feature_selection)}

shimmer_feature_selection = [
    'local',
    'localdb',
    'apq3',
    'apq5',
    'apq11',
    'dda',
] ## enable the ones you want
shimmer_feature_indices = {feature: idx for idx, feature in enumerate(shimmer_feature_selection)}

mfc_feature_selection = [    
    'mean', 
    'sd', 
    'skew', 
    'kurt', 
    'median', 
    'min', 
    'max', 
] ## enable the ones you want
mfc_feature_indices = {feature: idx for idx, feature in enumerate(mfc_feature_selection)}


