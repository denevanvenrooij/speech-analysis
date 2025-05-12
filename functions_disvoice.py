import importlib
import disvoice
importlib.reload(disvoice)

from disvoice.prosody import Prosody
from disvoice.phonation import Phonation
from disvoice.glottal import Glottal

def DV_prosody(audio_file):
    prosody = Prosody()
    prosody_features = prosody.extract_features_file(audio_file, static = True, plots=False, fmt="dataframe")
    
    return prosody_features

def DV_phonation(audio_file):
    phonation = Phonation()
    phonation_features = phonation.extract_features_file(audio_file, static = True, plots=False, fmt="dataframe")
    
    return phonation_features

def DV_glottal(audio_file):
    glottal = Glottal()
    glottal_features = glottal.extract_features_file(audio_file, static = True, plots=False, fmt="dataframe")
    
    return glottal_features
