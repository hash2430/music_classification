# GCT634 (2018) HW1 
#
# Mar-18-2018: initial version
# 
# Juhan Nam
#
# 코드가 굉장히 procedural programming으로 짜여 있다.
# 각 module을 run해야 한다.
# 코드가 procedural한 관계로, 시간 측정은 파일io 시간과 알고리즘 돌리는 시간을 구분하지 않는 것으로 했다

import os
import numpy as np
import librosa
from Baseline import util

data_path = '/home/sunghee/PycharmProjects/music_classification/example_dataset/'
mfcc_path = './mfcc/'

MFCC_DIM = 20

def extract_mfcc(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        #print file_path
        y, sr = librosa.load(file_path, sr=22050)


        ##### Method 1
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)

        ##### Method 2
        """
        # STFT
        S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1024)

        # power spectrum
        D = np.abs(S)**2

        # mel spectrogram (512 --> 40)
        mel_basis = librosa.filters.mel(sr, 1024, n_mels=40)
        mel_S = np.dot(mel_basis, D)

        #log compression
        log_mel_S = librosa.power_to_db(mel_S)

        # mfcc (DCT)
        mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13)
        mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)
        """

        # save mfcc as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = mfcc_path + file_name

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc)

    f.close();

if __name__ == '__main__':
    #training
    task = 'feature extraction'
    phase = 'train'
    util.start_measure(task + ' ' + phase)
    extract_mfcc(dataset=phase)
    util.finish_measure()

    #validation
    phase = 'validation'
    util.start_measure(task + ' ' + phase)
    extract_mfcc(dataset=phase)
    util.finish_measure()

    #test
    phase = 'test'
    util.start_measure(task + ' ' + phase)
    extract_mfcc(dataset=phase)
    util.finish_measure()



