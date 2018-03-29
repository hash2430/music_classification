import numpy as np
# experiment root path
method1_path = "/home/sunghee/PycharmProjects/music_classification/method1_is_baseline/"
method2_path = "/home/sunghee/PycharmProjects/music_classification/method2_svm_/"

# input data path
data_path = '/home/sunghee/PycharmProjects/music_classification/example_dataset/'

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
    return dir + mfcc_path + file_name.rstrip('\n')

def mean_mfcc_file(dir, phase):
    val = dir + mfcc_mean_path
    val += phase
    val += '_mfcc_mean.npy'
    return val

def mean_mfcc_visualization_file(dir, phase):
    return dir + mfcc_mean_visualization_path + phase + '_mfcc_visualization.png'

def report(dir):
    return dir + report_path + 'report'



data=('train', 'valid', 'test')

train_Y = np.array([1, 2, 3])
valid_Y = np.array([1, 2, 3])
test_Y = np.array([1, 2, 3])