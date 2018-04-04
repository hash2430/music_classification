import numpy as np
# experiment root path
method1_path = "/home/sunghee/PycharmProjects/music_classification/method1_baseline/"
method2_path = "/home/sunghee/PycharmProjects/music_classification/method2_feature_summary_nonlinear_svm/"
method3_path = "/home/sunghee/PycharmProjects/music_classification/method3_feature_extraction_nonlinear_svm/"

# input data path
data_path = '/home/sunghee/Documents/GCT634/HW1_dataset/dataset/'

# file paths relative to expriment root path
mfcc_path = 'mfcc/'
mfcc_mean_path = 'mfcc_mean/'
mfcc_mean_visualization_path = 'mfcc_mean_visualization/'
report_path = 'report/'

def feature_file_list(phase):
    return data_path + phase + '_list.txt'

def feature_file(file_name):
    return data_path + file_name.rstrip('\n')

def mfcc_file(dir, file_name):
    return dir + file_name.rstrip('\n')

def mean_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += '_mfcc_mean.npy'
    return val

def mean_mfcc_visualization_file(dir, phase):
    return dir + mfcc_mean_visualization_path + phase + '_mfcc_visualization.png'

def var_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += '_mfcc_var.npy'
    return val

def mean_var_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += '_mfcc_mean_var.npy'
    return val

def delta_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += '_mfcc_delta.npy'
    return val

def mean_var_delta_mean_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += '_mfcc_mean_var_delta.npy'
    return val


def flat_mfcc_file(dir, phase):
    val = dir
    val += phase
    val += 'flat_mfcc.npy'
    return val

data=('train', 'valid', 'test')

cls = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
train_Y = np.repeat(cls, 100)
valid_Y = np.repeat(cls, 20)
test_Y = np.repeat(cls, 20)
num_of_frames = 173