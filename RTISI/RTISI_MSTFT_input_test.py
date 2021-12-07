import librosa
import numpy as np
import soundfile as sf

from rtisi_functions import *

def main():
    
    file_name = "peixe-inteligente.wav"
    input_file = "../originais/"+file_name
    output_file = "../resultados/RTISI_"+file_name
    sr = 48000
    fft_size = 2048
    hop_size = fft_size // 4
    
    input_signal = load_signal(input_file, sr)
    n_frames = input_signal.size // hop_size
    
    # Minimum relative distance threshold:
    threshold = 0.01
    
    # Accumulate windows:
    windows_acc = np.empty( shape=(0,) )
    
    # Normalization step:
    gain = np.max(input_signal)
    input_signal = input_signal*1./gain
    
    # Perform zero padding at the end in case the signal doesn't fit the number
    # of FFT windows:
    if input_signal.size == (n_frames - 1)*hop_size + fft_size:
        yRTISI = np.zeros(input_signal.shape)
    else:
        yRTISI = np.zeros(input_signal.shape)
        yRTISI = np.pad(yRTISI, (0,((n_frames - 1)*hop_size + fft_size - input_signal.size)), 'constant')
    
    # Max number of itertions:
    iteration_max = 300
    
    Y = librosa.stft(input_signal, n_fft=fft_size, 
                     hop_length=hop_size, window='hann', 
                     center=False, pad_mode='reflect')
    
    Y = np.pad(Y, 
               ((0,0),
                (0,((n_frames - 1)*hop_size + fft_size - input_signal.size))), 
               'reflect')
    
    for frame in range(n_frames):
        iteration = 0
        
        # Accumulate windows according to frame:
        windows_acc = accumulate_windows(windows_acc, fft_size, hop_size, frame)
        
        # Variables to store 
        Dprev = 0
        Dcur = 0
        
        # Acumulates past distances to perform moving average
        memory = np.zeros(3)
        
        move_on = False
        
        while not move_on:
            
            # In the first iteration, the recovered signal's FFT is defined as
            # having the same magnitude as the input signal's FFT and phase
            # identical to zero
            if iteration == 0: 
                Y_it = Y[:,frame]
                start = True
            else:
                memory[0] = memory[1]
                memory[1] = memory[2]
                memory[2] = Dcur
                Dprev = np.sum(memory)/memory.size
                start = False
                
            Y_mag = np.abs(Y_it)
            
            if iteration == 0: 
                Y_it = Y_mag
                Dcur = 1000
            else:
                X = frame_fft(x, fft_size, hop_size, frame)
                Y_it, Dcur = update_Yfft(X, Y_mag, start)
            
            y = ifft(Y_it, fft_size)
            
            x = update_x(y, yRTISI, fft_size, hop_size, frame, windows_acc)
            
            x_win, move_on = x_eval(x, threshold, Dcur, Dprev, fft_size, hop_size, frame, start)
            
            iteration += 1
            
            #print("frame {0} of {1}: iteration {2}".format(frame+1,n_frames,iteration))
            
            if iteration >= iteration_max:
                print("frame {0} of {1} used max iterations".format(frame+1,n_frames))
                break
            
        yRTISI = update_yRTISI(yRTISI, x_win, fft_size, hop_size, frame)
        #print(yRTISI[frame])
    
    to_be_saved = yRTISI[:input_signal.size]
    save_signal(gain*to_be_saved/np.max(to_be_saved), output_file, sr)
    
if __name__ == '__main__':
    main()