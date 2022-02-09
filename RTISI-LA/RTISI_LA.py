import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import soundfile as sf

def RTISI_LA(Y_mag, FS, fft_size, w_size, hop_size, LA, thresh):
    
    out_file = './resultados/RTISI_LA_out.wav'
    
    # --- Salva o numero de frames:
    n_frames = Y_mag.shape[1]
    
    # --- Realiza zero padding para tratar dos últimos frames:
    Y_mag = np.pad(Y_mag, ((0, 0), (0, 2*LA)), mode = 'reflect')
    
    # --- Protótipo do espectro da saida:
    END_SPEC = np.zeros(Y_mag.shape)
    
    # --- Inicializacoes em zero:
    y_RTISI = np.zeros((n_frames+LA+1)*hop_size + fft_size)
    x = np.zeros((n_frames+LA+1)*hop_size + fft_size)
    windows_acc = np.zeros((n_frames+LA+1)*hop_size + fft_size)
    
    # --- Define numericamente a janela:
    window = np.zeros(fft_size)
    window[:w_size] += np.hamming(w_size)
    
    print('Inicio: RTISI-LA para {0} quadros'.format(n_frames))
    
    max_iter = 0
    
    for frame in range(n_frames):
        
        y = np.zeros((LA+1,window.size))
        # --- Inicializacao:
        for i in range(LA + 1):
            y[i,:] = np.fft.irfft(Y_mag[:,frame+i]*np.exp(1j*0))
        
        begin = hop_size*frame
        end = begin + fft_size
        if frame == 0:
            for i in range(LA + 1):
                windows_acc[begin+i*hop_size:end+i*hop_size] += window**2
        else:
            windows_acc[begin+(LA)*hop_size:end+(LA)*hop_size] += window**2
        
        x = np.zeros((n_frames+LA+1)*hop_size + fft_size)
        for i in range(LA+1):
            x[begin+i*hop_size:end+i*hop_size] += window*y[i]
        x[:end+LA*hop_size] = (x[:end+LA*hop_size] + y_RTISI[:end+LA*hop_size])/windows_acc[:end+LA*hop_size]
        
        x_win = np.zeros((LA+1, window.size))
        X_win = np.zeros((LA+1, Y_mag.shape[0]))*np.exp(1j*0)
        for i in range(LA + 1):
            x_win[i,:] = window*x[begin+i*hop_size:end+i*hop_size]
            X_win[i,:] = np.fft.rfft(x_win[i,:])
        
        Dcur = np.sum((np.abs(X_win[0,:]) - Y_mag[:,frame])**2)
        
        Dprev = Dcur + 1000
        
        # --- Iteracoes segundo o diagrama de blocos:
        iterc = 0
        while np.abs(Dprev - Dcur)/Dprev > thresh:
            
            # IFFT
            for i in range(LA + 1):
                y[i,:] = np.fft.irfft(Y_mag[:,frame+i]*np.exp(1j*np.angle(X_win[i,:])))
            
            # Formula de atualizacao:
            x = np.zeros((n_frames+LA+1)*hop_size + fft_size)
            for i in range(LA+1):
                x[begin+i*hop_size:end+i*hop_size] += window*y[i]
            x[:end+LA*hop_size] = (x[:end+LA*hop_size] + y_RTISI[:end+LA*hop_size])/windows_acc[:end+LA*hop_size]

            # Multiplicacao pela janela de analise:
            for i in range(LA + 1):
                x_win[i,:] = window*x[begin+i*hop_size:end+i*hop_size]
                #FFT
                X_win[i,:] = np.fft.rfft(x_win[i,:])            
            
            # Atualizacao das distancias:
            Dprev = Dcur
            Dcur = np.sum((np.abs(X_win[0,:]) - np.abs(Y_mag[:,frame]))**2)
            
            iterc += 1
            
        if iterc > max_iter:
            max_iter = iterc
            
        # --- Acumulo do resultado:
        y_RTISI[begin:end] += window*x_win[0,:]
        END_SPEC[:,frame] = np.abs(X_win[0,:])
    
    plt.plot(windows_acc[:end])
    
    # Corrigindo o erro de janelamento no resultado acumulado:
    y_reconstruido = y_RTISI[:end]/np.sqrt(windows_acc[:end])
    
    print('Fim RTISI-LA, maximo num. de iteracoes: {0}'.format(max_iter))
    
    sf.write(out_file, y_reconstruido/np.max(np.abs(y_reconstruido)), FS)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(END_SPEC, ref=np.max), y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(Y_mag, ref=np.max), y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")  
    