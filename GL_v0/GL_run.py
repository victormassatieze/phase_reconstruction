import librosa
import numpy as np
import soundfile as sf

from GL_functions import *

def run_GL(file_name, fft_size, hop_size, in_gain = 1):
    
    #file_name = "peixe-inteligente.wav"
    #fft_size = 2048
    #hop_size = fft_size // 4
    
    input_file = "./originais/"+file_name
    output_file = "./resultados/fft2048/GL_"+file_name
    sr = 48000
    
    input_signal = load_signal(input_file, sr)
    
    # Normalization step:
    gain = np.max(input_signal)
    input_signal = input_signal*1./gain
    
    sig_len = input_signal.size
    
    # Minimum relative distance threshold:
    thresh = 0.01
    
    print("GL: start; file: {0}".format(file_name))
    
    initial_spectrum = librosa.stft(input_signal, n_fft=fft_size, 
                                      hop_length=hop_size, window='hann', 
                                      center=False, pad_mode='reflect') #Yw
    #recovered_spectrum = np.pad(Y, 
    #           ((0,0),
    #            (0,((n_frames - 1)*hop_size + fft_size - input_signal.size))), 
    #           'reflect')
    
    recovered_spectrum = np.abs(initial_spectrum)*np.exp(1j*0)
    
    opt_win = 'hann'
    
    proposed_signal = get_windowed_signal(recovered_spectrum, hop_size,
                                          fft_size, sig_len, opt_win)
    
    proposed_spectrum = librosa.stft(proposed_signal, n_fft=fft_size, 
                                     hop_length=hop_size, window='hann', 
                                     center=False, pad_mode='reflect') 
    
    D, move_on = get_distance(recovered_spectrum, proposed_spectrum)
    
    iteration = 0
    
    while not move_on:
        
        iteration += 1
        
        #print("Iteration {0}, Euclidian distance: {1}".format(iteration,D))
        
        recovered_spectrum = np.abs(initial_spectrum)*np.exp(1j*np.angle(proposed_spectrum))
        
        proposed_signal = get_windowed_signal(recovered_spectrum, hop_size,
                                              fft_size, sig_len, opt_win)
        
        proposed_spectrum = librosa.stft(proposed_signal, n_fft=fft_size, 
                                         hop_length=hop_size, window='hann', 
                                         center=False, pad_mode='reflect')
        
        D, move_on = get_distance(recovered_spectrum, proposed_spectrum, D, thresh)        
        
    
    to_be_saved = proposed_signal[:input_signal.size]
    save_signal(in_gain*gain*to_be_saved/np.max(to_be_saved), output_file, sr)
    
    print("GL done")
    
#if __name__ == '__main__':
#    main()