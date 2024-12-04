import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft
import librosa  as lb
from glob import glob
#import IPython.display as ipd
from pathlib import Path
import sounddevice as sd

def performFFT(signal, srate, freq_cutoff = 1000):
    n = len(signal)
    #Find next power of 2 that is larger than the signal length
    #then perform FFT
    nfft = int(2**(np.ceil(np.log2(n))))
    signal_fft = fft(signal,n=nfft,norm='ortho')
    #Return one-sided FFT.
    half_signal=int(np.ceil(nfft/2))
    signal_fft=signal_fft[0:half_signal+1]
    freqs = lb.fft_frequencies(sr=srate,n_fft=nfft)
    #Report frequencies below cutoff
    cutoff = np.where(freqs < freq_cutoff)
    cut_freqs = freqs[cutoff]
    cut_signal = signal_fft[cutoff]
    return cut_signal, cut_freqs

#Read C_1.wav file
file = "./data_pure_tones/C_1.wav"
Fs=44100
freq_cutoff = 1000
audio_data, Fs = lb.load(file,sr=Fs)
sd.play(audio_data,Fs)

yfft, freqs = performFFT(audio_data, Fs, freq_cutoff)
plt.plot(freqs,np.abs(yfft))
plt.show()

