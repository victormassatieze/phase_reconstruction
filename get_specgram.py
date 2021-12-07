import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display

def save_spectrograms(file_name, sr, opt = 'linear'):

    files = ["", "", "", ""]
    count = 0
    for i in ["GL_", "RTISI_", "RTISI_LA_"]:
        files[count] = "./resultados/"+i+file_name+".wav"
        count += 1
    files[count] = "./originais/"+file_name+".wav"
    
    types = ["GL_", "RTISI_", "RTISI_LA_", "0_original_"]
    count = 0

    for file in files:
        y, sr = librosa.load(file, sr)
            
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        plt.figure(figsize=(16, 8))
        librosa.display.specshow(S_db,y_axis=opt, x_axis='time', sr=sr)
        plt.title(opt+'-frequency power spectrogram of '+types[count]+file_name)
        plt.colorbar()
        plt.savefig("./resultados/specgrams/"+types[count]+file_name+".png")
        
        count += 1