import numpy as np
#from numpy.fft import fft, ifft
#from matplotlib import pyplot as plt
import pandas as pd
import librosa  as lb
#import librosa.display as lbd
from glob import glob
import re

class AudioFeatureExtractor():
    def __init__(self,data_dir):
        self.audio_files = glob(data_dir+'/*.wav')
        
    def getTargetLabel(self, file):
        #filename="".join(re.split('./data',file))
        filename = file.split("/")[-1]
        target = filename[0]
        return target

    def splitSignal(self, audioSignal, nsegments):
        signalLength = len(audioSignal)
        segmentLength = int(np.ceil(signalLength/nsegments))
        audioSegments = []
        for i in range(nsegments):
            first = i*segmentLength
            last = first  + segmentLength - 1
            if last > (signalLength-1):
                last  = signalLength-1
            audioSegments.append(audioSignal[first:last])
    
        return audioSegments
    
    def constructMFCCFeatures(self, nsegments=10, num_mfcc=20):
        column_labels=["Target"]
        for q in range(num_mfcc):
            column_labels.append("MFCC "+str(q))
        data = []

        for file in self.audio_files:
            audio_data, Fs = lb.load(file,sr=44100)
            segments = self.splitSignal(audio_data, nsegments)
            target = self.getTargetLabel(file)
            for j in range(nsegments):
                #D = lb.stft(data)
                D= lb.feature.mfcc(y=segments[j],sr=Fs, n_mfcc=num_mfcc)
                s_db = np.mean(lb.amplitude_to_db(np.abs(D),ref = np.max),1)
                data_entry = [target] + s_db.tolist()
                data.append(data_entry)
        self.mfcc_data_frame = pd.DataFrame(data, columns = column_labels)
        return self.mfcc_data_frame
       

# plt.figure()
# lbd.specshow(s_db, x_axis='time', y_axis='log', sr=Fs, fmin=4, fmax=8)
# plt.colorbar(format="%+2.f dB")