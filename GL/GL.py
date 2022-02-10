import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import soundfile as sf

def GL(Y_mag, FS, fft_size, w_size, hop_size, thresh):
    
    out_file = './resultados/GL_out.wav'
    
    # --- Salva o numero de frames:
    n_frames = Y_mag.shape[1]
    
    # --- Protótipo do espectro da saida:
    END_SPEC = np.zeros(Y_mag.shape)
    
    # --- Inicializacoes em zero:
    propose = np.zeros((n_frames - 1)*hop_size + fft_size)
    proposed_SPEC = np.zeros(Y_mag.shape)*np.exp(1j*0)
    windows_acc_init = np.zeros((n_frames - 1)*hop_size + fft_size + w_size)
    
    # --- Define numericamente a janela:
    window = np.zeros(fft_size)
    window[:w_size] += np.hanning(w_size)
    
    # --- Acumula as janelas:
    for frame in range(n_frames + fft_size//hop_size):
        begin = hop_size*frame
        end = begin + fft_size
        windows_acc_init[begin:end] += window**2
        
    windows_acc = windows_acc_init[w_size//2: w_size//2 + propose.size]
    
    print('Inicio: GL para {0} quadros'.format(n_frames))
    
    iterc = 0
    
    Dcur = 1    
    Dprev = Dcur + 1000
    
    while np.abs(Dprev - Dcur)/Dprev > thresh:
    
        # constroi o buffer a ser janelado:
        for frame in range(n_frames):
            
            begin = hop_size*frame
            end = begin + fft_size
            
            if iterc == 0:
                y = np.fft.irfft(Y_mag[:,frame]*np.exp(1j*0))
            else:
                y = np.fft.irfft(Y_mag[:,frame]*np.exp(np.angle(proposed_SPEC[:,frame])))
                
            propose[begin:end] += window*y
            
        # corrige erro de janelamento:
        propose = propose/windows_acc
        
        for frame in range(n_frames):
            
            begin = hop_size*frame
            end = begin + fft_size
            
            proposed_SPEC[:,frame] = np.fft.rfft(window*propose[begin:end])
        
        Dprev = Dcur
        Dcur = np.sum((np.abs(proposed_SPEC) - Y_mag)**2)/Y_mag.size
        print(Dcur)
        
        iterc += 1
    
    print('Fim GL, num. de iteracoes: {0}'.format(iterc))
    
    sf.write(out_file, propose/np.max(np.abs(propose)), FS)