# input: mfcc files
# output: mean mfcc files, variance mfcc files, spectrogram image files

import librosa
import os
import numpy as np
from NotRunnables import Path
import NotRunnables.FeatureExtraction as FeatureExtraction
#from NotRunnables import *
import matplotlib.pyplot as plt

# for visualization & to be used as feature
# save as file
class FeatureSummary():
    inputDir = ""
    outputDir = ""

    def __init__(self, mfcc_dim, inputDir, outputDir):
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.mfcc_dim = mfcc_dim

    #todo: 데이터 크기를 미리 Path에 정의해놓자
    def mean_mfcc(self, n):
        phase = Path.data[n]
        list_file = Path.feature_file_list(phase)
        f = open(list_file, 'r')
        sample_size = sum(1 for line in f)
        f.close()
        f = open(list_file, 'r')
        mfcc_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        #todo: progress check line을 프린트할까(i 변수 사용)
        i = 0

        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = Path.mfcc_file(self.inputDir + phase + '/', file_name)
            mfcc = np.load(mfcc_file)

            # mean pooling
            # 각 파일의 시간 축에서의 평균
            temp = np.mean(mfcc, axis=1)
            mfcc_mat[:, i] = temp
            i = i + 1

        f.close();
        save_file = Path.mean_mfcc_file(self.outputDir, phase)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc_mat)
        return mfcc_mat #(mfcc_dim, num of files)

    # spectrogram visualization function
    # todo: 그림 이쁘게 그리기
    def visualize(self, n):
        phase = Path.data[n]
        mean_mfcc_file = Path.mean_mfcc_file(self.outputDir, phase)
        mean_mfcc = np.load(mean_mfcc_file)
        save_file = Path.mean_mfcc_visualization_file(self.outputDir, phase)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        plt.figure(1)
        plt.imshow(mean_mfcc)
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(save_file)

    def var_mfcc(self, n):
        phase = Path.data[n]
        list_file = Path.feature_file_list(phase)
        f = open(list_file, 'r')
        sample_size = sum(1 for line in f)
        f.close()
        f = open(list_file, 'r')
        mfcc_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        #todo: progress check line을 프린트할까(i 변수 사용)
        i = 0

        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = Path.mfcc_file(self.inputDir + phase + '/', file_name)
            mfcc = np.load(mfcc_file)

            # mean pooling
            # 각 파일의 시간 축에서의 평균
            temp = np.std(mfcc, axis=1)
            mfcc_mat[:, i] = temp
            i = i + 1

        f.close();
        save_file = Path.var_mfcc_file(self.outputDir, phase)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc_mat)
        return mfcc_mat #(mfcc_dim, num of files)

    def delta_mfcc(self, n):
        phase = Path.data[n]
        list_file = Path.feature_file_list(phase)
        f = open(list_file, 'r')
        sample_size = sum(1 for line in f)
        f.close()
        f = open(list_file, 'r')
        mfcc_delta_mean_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        mfcc_delta_var_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        mfcc_delta_delta_mean_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        mfcc_delta_delta_var_mat = np.zeros(shape=(self.mfcc_dim, sample_size))
        mfcc_mat = np.zeros(shape=(self.mfcc_dim * 4, sample_size))
        #todo: progress check line을 프린트할까(i 변수 사용)
        i = 0

        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = Path.mfcc_file(self.inputDir + phase + '/', file_name)
            mfcc = np.load(mfcc_file)

            delta = librosa.feature.delta(mfcc)
            delta_mean = np.mean(delta, 1)
            mfcc_delta_mean_mat[:, i] = delta_mean#(60,1000)
            delta_var = np.std(delta, 1)
            mfcc_delta_var_mat[:, i] = delta_var

            delta_delta = librosa.feature.delta(mfcc, order=2)
            delta_delta_mean = np.mean(delta_delta, 1)
            mfcc_delta_delta_mean_mat[:, i] = delta_delta_mean#(60,1000)
            delta_delta_var = np.std(delta_delta, 1)
            mfcc_delta_delta_var_mat[:, i] = delta_delta_var
            i = i + 1
        f.close();
        mfcc_mat = np.concatenate((mfcc_delta_mean_mat, mfcc_delta_var_mat, mfcc_delta_delta_mean_mat, mfcc_delta_delta_var_mat), axis=0)
        save_file = Path.delta_mfcc_file(self.outputDir, phase)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc_mat)
        return mfcc_mat #(mfcc_dim, num of files)

    def flatten(self, n):
        phase = Path.data[n]
        list_file = Path.feature_file_list(phase)
        f = open(list_file, 'r')
        sample_size = sum(1 for line in f)
        f.close()
        f = open(list_file, 'r')
        mfcc_mat = np.zeros(shape=(self.mfcc_dim * Path.num_of_frames, sample_size)) #(mfcc_dim*frame_num, sample_size)
        # todo: progress check line을 프린트할까(i 변수 사용)
        i = 0

        for file_name in f:
            # load mfcc file
            file_name = file_name.rstrip('\n')
            file_name = file_name.replace('.wav', '.npy')
            mfcc_file = Path.mfcc_file(self.inputDir + phase + '/', file_name)
            mfcc = np.load(mfcc_file)

            # mean pooling
            # 각 파일의 시간 축에서의 평균
            temp = mfcc.flatten('C')
            mfcc_mat[:, i] = temp
            i = i + 1

        f.close();
        save_file = Path.flat_mfcc_file(self.outputDir, phase)
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, mfcc_mat)
        return mfcc_mat  # (mfcc_dim, num of files)



class FeatureConcatenator():
    input1_dir = ""
    input2_dir = ""

    def __init__(self):
        pass
    def concat(self, input1, input2, save_file):
        # load training feature, validation feature
        input1 = np.load(input1)
        input2 = np.load(input2)

        output = np.concatenate((input1, input2), axis=0)

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, output)
        return output #(mfcc_dim * 2, num of files)
