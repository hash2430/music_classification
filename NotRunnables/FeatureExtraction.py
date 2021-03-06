# input: wav file, list of wav file names
# output: mfcc feature file for each wav
#ToDo: class가 dir 멤버를 constructor로부터 받아서 사용하도록 고치기
#ToDo: phase를 직접 입력받지 않ㄱ소 Path 모듈의 파라미터를 참조하게 하기
from NotRunnables import  Path

import librosa
import os
import numpy as np
#from NotRunnables import *

class FeatureExtraction():
    outputDir = ""
    sr = 22050
    mfcc_dim = 20

    def __init__(self, outputDir, sr=22050, mfcc_dim=20):
        self.outputDir = outputDir
        self.sr = sr
        self.mfcc_dim = mfcc_dim

    def extract_mfcc1(self, phase):
        phase = Path.data[phase]
        list_file = Path.feature_file_list(phase)
        f = open(list_file, 'r')

        i = 0
        for file_name in f:
            # progress check
            i = i + 1
            if not (i % 10):
                print(i)

            # load audio file
            file_path = Path.feature_file(file_name)
            # print file_path
            y, sr = librosa.load(file_path, sr=self.sr)

            ##### Method 1
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.mfcc_dim)

            # save mfcc as a file
            file_name = file_name.replace('.wav', '.npy')
            save_file = Path.mfcc_file(self.outputDir + phase + '/', file_name)

            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, mfcc)

        f.close();

    def extract_mfcc2(self, phase):
        f = open(Path.feature_file_list(Path.data[phase]), 'r')

        i = 0
        for file_name in f:
            # progress check
            i = i + 1
            if not (i % 10):
                print(i)

            # load audio file
            file_path = Path.feature_file(file_name)

            # print file_path
            y, sr = librosa.load(file_path, sr=self.sr)

            ##### Method 2
            # STFT
            S = librosa.core.stft(y, n_fft=1024, hop_length=512, win_length=1023)

            # power spectrum
            D = np.abs(S)**2

            # mel spectrogram (512 --> 40)
            mel_basis = librosa.filters.mel(sr, 1024, n_mels=256)
            mel_S = np.dot(mel_basis, D)

            #log compression
            log_mel_S = librosa.power_to_db(mel_S)

            # mfcc (DCT)
            mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=60)
            mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)

            # save mfcc as a file
            file_name = file_name.replace('.wav', '.npy')
            save_file = Path.mfcc_file(self.outputDir + Path.data[phase] + '/', file_name)
            dir_part = os.path.dirname(save_file)
            if not os.path.exists(dir_part):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, mfcc)

        f.close();
