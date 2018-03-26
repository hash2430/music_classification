# input: mfcc files
# output: mean mfcc files, variance mfcc files, spectrogram image files

import librosa
import os
import numpy as np
import Path
import matplotlib.pyplot as plt
from FeatureExtraction import MFCC_DIM

# for visualization & to be used as feature
# save as file
class FeatureSummary():
    dir = ""

    def __init__(self, dir):
        self.dir = dir

    def mean_mfcc(self, n):
        phase = Path.data[n]
        f = open(Path.data_path + phase + '_list.txt', 'r')
        sample_size = sum(1 for line in f)
        mfcc_mat = np.zeros(shape=(MFCC_DIM, sample_size))

        i = 0
        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = self.dir + Path.mfcc_path + file_name
            mfcc = np.load(mfcc_file)

            # mean pooling
            # 각 파일의 시간 축에서의 평균
            temp = np.mean(mfcc, axis=1)
            mfcc_mat[:, i] = temp
            i = i + 1

        f.close();
        save_file = self.dir + Path.mfcc_mean_path + phase + '_mfcc_mean.npy'
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc_mat)
        return mfcc_mat

    # spectrogram visualization function
    def visualize(self, n):
        phase = Path.data[n]
        mean_mfcc_file = self.dir + Path.mfcc_mean_path + phase + '_mfcc_mean.npy'
        mean_mfcc = np.load(mean_mfcc_file)
        save_file = self.dir + Path.mfcc_mean_visualization_path + phase + '_mfcc_visualization.png'
        plt.figure(1)
        plt.imshow(mean_mfcc)
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(save_file)