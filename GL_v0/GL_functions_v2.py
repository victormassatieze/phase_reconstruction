'''
Base functions for RTISI-LA

Last Update: window accumulation function
'''

import librosa
import numpy as np
import soundfile as sf

def load_signal(file, sr_check = 44100):
    y, sr = librosa.load(file, sr_check, dtype = np.float32)
    
    if y.ndim == 1:
        return y
    else:
        return y.mean(axis=1)

def get_windowed_signal(recovered_spectrum, hop_size, fft_size, signal_len, opt_win = 'hann'):
    assert fft_size == int(fft_size)
    assert hop_size == int(hop_size)
    
    if opt_win == 'hann':
        window = np.hanning(fft_size);
    else:
        raise Exception("Unkown window.")
    
    n_frames = recovered_spectrum.shape[1]
    
    proposed_signal = np.zeros(signal_len)
    window_acc = np.zeros(signal_len)
    
    for frame in range(n_frames):
        #print("Frame {0}".format(frame))
        
        begin = frame*(hop_size)
        end = begin + fft_size
        proposed_signal[begin:end] += window*np.real(np.fft.ifft(recovered_spectrum[:,frame],fft_size))
        
    return proposed_signal

def get_distance(X, Y, Dprev = None, thresh = None):
    diff = (np.abs(X) - np.abs(Y))**2
    #print("X: {0}; Y: {1}; Difference: {2}".format(X,Y,diff))
    D = np.sum(diff)/diff.size
    
    if not Dprev:
        return D, False
    
    # Evaluate Euclidian distance:
    if (Dprev - D)/Dprev < thresh:
        return D, True
    
    return D, False

def save_signal(y, file, sr):
    sf.write(file, y, sr)