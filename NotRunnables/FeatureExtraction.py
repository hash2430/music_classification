# input: wav file, list of wav file names
# output: mfcc feature file for each wav

import librosa
import os
import numpy as np
from Path import data_path, mfcc_path

MFCC_DIM = 20
class FeatureExtraction():
    def extract_mfcc1(dataset='train'):
        f = open(data_path + dataset + '_list.txt', 'r')

        i = 0
        for file_name in f:
            # progress check
            i = i + 1
            if not (i % 10):
                print(i)

            # load audio file
            file_name = file_name.rstrip('\n')
            file_path = data_path + file_name
            # print file_path
            y, sr = librosa.load(file_path, sr=22050)

            ##### Method 1
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_DIM)

            # save mfcc as a file
            file_name = file_name.replace('.wav', '.npy')
            save_file = mfcc_path + file_name

            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, mfcc)

        f.close();

    def extract_mfcc2(dataset='train'):
        f = open(data_path + dataset + '_list.txt', 'r')

        i = 0
        for file_name in f:
            # progress check
            i = i + 1
            if not (i % 10):
                print(i)

            # load audio file
            file_name = file_name.rstrip('\n')
            file_path = data_path + file_name

            # print file_path
            y, sr = librosa.load(file_path, sr=22050)

            ##### Method 2
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

            # save mfcc as a file
            file_name = file_name.replace('.wav', '.npy')
            save_file = mfcc_path + file_name

            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, mfcc)

        f.close();
