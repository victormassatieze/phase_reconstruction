import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import soundfile as sf

def RTISI(Y_mag, FS, fft_size, w_size, hop_size, thresh):
    
    out_file = './resultados/RTISI_out.wav'
    
    # --- Salva o numero de frames:
    n_frames = Y_mag.shape[1]
    
    # --- Protótipo do espectro da saida:
    END_SPEC = np.zeros(Y_mag.shape)
    
    # --- Inicializacoes em zero:
    y_RTISI = np.zeros(n_frames*hop_size + fft_size)
    x = np.zeros(n_frames*hop_size + fft_size)
    windows_acc = np.zeros(n_frames*hop_size + fft_size)
    
    # --- Define numericamente a janela:
    window = np.zeros(fft_size)
    window[:w_size] += np.hamming(w_size)
    
    print('Inicio: RTISI para {0} quadros'.format(n_frames))
    
    max_iter = 0
    
    for frame in range(n_frames):
        
        # --- Inicializacao:
        y = np.fft.irfft(Y_mag[:,frame]*np.exp(1j*0))
        
        begin = hop_size*frame
        end = begin + fft_size
        windows_acc[begin:end] += window**2
        
        x = np.zeros(n_frames*hop_size + fft_size)
        x[begin:end] += window*y
        x[:end] = (x[:end] + y_RTISI[:end])/windows_acc[:end]
        
        x_win = window*x[begin:end]
        
        X_win = np.fft.rfft(x_win)
        
        Dcur = np.sum((np.abs(X_win) - Y_mag[:,frame])**2)
        
        Dprev = Dcur + 1000
        
        # --- Iteracoes segundo o diagrama de blocos:
        iterc = 0
        while np.abs(Dprev - Dcur)/Dprev > thresh:
            
            # IFFT
            y = np.fft.irfft((Y_mag[:,frame])*np.exp(1j*np.angle(X_win)))
            
            # Formula de atualizacao:
            x = np.zeros(n_frames*hop_size + fft_size)
            x[begin:end] += window*y
            x[:end] = (x[:end] + y_RTISI[:end])/windows_acc[:end]

            # Multiplicacao pela janela de analise:
            x_win = window*x[begin:end]

            # FFT:
            X_win = np.fft.rfft(x_win)
            
            # Atualizacao das distancias:
            Dprev = Dcur
            Dcur = np.sum((np.abs(X_win) - np.abs(Y_mag[:,frame]))**2)
            
            iterc += 1
            
        if iterc > max_iter:
            max_iter = iterc
            
        # --- Acumulo do resultado:
        y_RTISI[begin:end] += window*x_win
        
    plt.plot(windows_acc[:end])
    
    # Corrigindo o erro de janelamento no resultado acumulado:
    y_reconstruido = y_RTISI[:end]/np.sqrt(windows_acc[:end])
    
    print('Fim RTISI, maximo num. de iteracoes: {0}'.format(max_iter))
    
    sf.write(out_file, y_reconstruido/np.max(np.abs(y_reconstruido)), FS)  
    