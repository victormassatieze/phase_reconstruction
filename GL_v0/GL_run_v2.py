import librosa
import numpy as np
import soundfile as sf

from GL_functions import *

def run_GL(MAG_spec, sr, fft_size, hop_size, window = 'hann', output_file = 'GL_result.wav'):
    
    # Minimum relative distance threshold:
    thresh = 0.01
    
    sig_len = MAG_spec.shape[1]*hop_size + (fft_size - hop_size)
    
    initial_spectrum = MAG_spec #Yw
    
    recovered_spectrum = np.abs(initial_spectrum)*np.exp(1j*0)
    
    opt_win = 'hann'
    
    proposed_signal = get_windowed_signal(recovered_spectrum, hop_size,
                                          fft_size, sig_len, opt_win)
    
    proposed_spectrum = librosa.stft(proposed_signal, n_fft=fft_size, 
                                     hop_length=hop_size, window=window, 
                                     center=False, pad_mode='reflect') 
    
    D, move_on = get_distance(recovered_spectrum, proposed_spectrum)
    
    iteration = 0
    
    while not move_on:
        
        iteration += 1
        
        recovered_spectrum = np.abs(initial_spectrum)*np.exp(1j*np.angle(proposed_spectrum))
        
        proposed_signal = get_windowed_signal(recovered_spectrum, hop_size,
                                              fft_size, sig_len, opt_win)
        
        proposed_spectrum = librosa.stft(proposed_signal, n_fft=fft_size, 
                                         hop_length=hop_size, window='hann', 
                                         center=False, pad_mode='reflect')
        
        D, move_on = get_distance(recovered_spectrum, proposed_spectrum, D, thresh)        
        
    
    to_be_saved = proposed_signal
    save_signal(to_be_saved, output_file, sr)
    
    print("GL done")
    
if __name__ == '__main__':
    
    fft_size = 2048
    hop_size = fft_size//4
    win = 'hann'

    signal, sr = librosa.load('../originais/x1.wav')

    MAG_spec = librosa.stft(signal, n_fft=fft_size, 
                            hop_length=hop_size, window=win, 
                            center=False, pad_mode='reflect')

    run_GL(MAG_spec, sr, fft_size, hop_size, window = win, output_file = 'GL_x1.wav')