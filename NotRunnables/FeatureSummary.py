# input: mfcc files
# output: mean mfcc files, variance mfcc files, spectrogram image files

import librosa
import os
import numpy as np
from Path import data_path, mfcc_path
from FeatureExtraction import MFCC_DIM

# for visualization & to be used as feature
# save as file
class FeatureSummary():
    def mean_mfcc(dataset='train'):
        f = open(data_path + dataset + '_list.txt', 'r')

        if dataset == 'train':
            mfcc_mat = np.zeros(shape=(MFCC_DIM, 1000))
        else:
            mfcc_mat = np.zeros(shape=(MFCC_DIM, 200))

        i = 0
        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = mfcc_path + file_name
            mfcc = np.load(mfcc_file)

            # mean pooling
            temp = np.mean(mfcc, axis=1)
            mfcc_mat[:, i] = np.mean(mfcc, axis=1)
            i = i + 1

        f.close();

        return mfcc_mat

    # spectrogram visualization function