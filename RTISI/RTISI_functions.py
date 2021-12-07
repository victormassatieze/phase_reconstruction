'''
Base functions for RTISI

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

def frame_fft(x, fft_size, hop_size, frame, opt_win = "hanning"):
    assert fft_size == int(fft_size)
    assert hop_size == int(hop_size)
    assert frame == int(frame)
    
    if opt_win == "hanning":
        window = np.hanning(int(fft_size))
    else:
        raise Exception("Unkown window.")
        
    begin = frame*(hop_size)
    ender = begin + fft_size
    
    if window.shape == x[begin:ender].shape:
        return np.fft.rfft(window*x[begin:ender])
    else:
        x_aux = x[begin:]
        x_aux = np.pad(x_aux, (0,(ender-x.size)), 'constant')
        return np.fft.rfft(window*x_aux)

def ifft(X, fft_size):
    assert fft_size == int(fft_size)
    
    return np.fft.irfft(X)

def get_distance(X, Y):
    diff = (np.abs(X) - np.abs(Y))**2
    #print("X: {0}; Y: {1}; Difference: {2}".format(X,Y,diff))
    D = np.sum(diff)/diff.size
    return D

def update_Yfft(X, Y_mag, start = False):
    if not start:
        Y = Y_mag*np.exp(1.0j*np.angle(X))
    else:
        Y = Y_mag*np.exp(1.0j*0)
    
    Dcur = get_distance(X, Y)
    return Y, Dcur

def accumulate_windows(windows_acc, fft_size, hop_size, frame, opt_win = "hanning"):
    assert frame == int(frame)
    
    if opt_win == "hanning":
        window = np.hanning(int(fft_size))
        #print(window.shape)
    else:
        raise Exception("Unkown window.")
        
    if frame == 0: new_accumulation = np.append(windows_acc,window**2)
    else:
        new_accumulation = windows_acc
        new_accumulation[new_accumulation.size - (fft_size-hop_size):] += window[:fft_size-hop_size]**2
        new_accumulation = np.append(new_accumulation, window[fft_size-hop_size:]**2)
    
    return new_accumulation
    
def update_x(y, yRTISI, fft_size, hop_size, frame, windows_acc, opt_win = "hanning"):
    assert fft_size == int(fft_size)
    assert hop_size == int(hop_size)
    assert frame == int(frame)
    
    if opt_win == "hanning":
        window = np.hanning(int(fft_size))
        #print(window.shape)
    else:
        raise Exception("Unkown window.")
        
    aux = yRTISI
    x = np.zeros(aux.shape)
    
    if frame != 0:        
        begin = frame*(hop_size)
        ender = begin + fft_size
        aux[begin:ender] += window*y
        x[:begin+hop_size] = aux[:begin+hop_size]
        x[begin+hop_size:ender] = aux[begin+hop_size:ender]
        x = x/np.sum(windows_acc)
    else:
        x = window*y/np.sum(window**2)
    
    return x

def x_eval(x, threshold, Dcur, Dprev, fft_size, hop_size, frame, start = False, opt_win = "hanning"):
    assert fft_size == int(fft_size)
    assert hop_size == int(hop_size)
    assert frame == int(frame)
    
    if opt_win == "hanning":
        window = np.hanning(int(fft_size))
    else:
        raise Exception("Unkown window.")
    
    x_win = np.zeros(window.shape)
    
    begin = frame*(hop_size)
    ender = begin + fft_size
    x_win = window*x[begin:ender]
    
    if not start:
        if ((Dprev - Dcur)/Dprev < threshold):
            move_on = True
        else:
            move_on = False
    else:
        move_on = False
        
    return x_win, move_on

def update_yRTISI(yRTISI, x_win, fft_size, hop_size, frame):
    assert fft_size == int(fft_size)
    assert hop_size == int(hop_size)
    assert frame == int(frame)
    
    begin = frame*(hop_size)
    ender = begin + fft_size
    yRTISI[begin:ender] += x_win
    
    return yRTISI
         
def save_signal(y, file, sr):
    sf.write(file, y, sr)
    
    

