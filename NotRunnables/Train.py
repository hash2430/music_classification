# input: mean mfcc files, variance mfcc files
# output: hyperparameters and models files
from NotRunnables import Path, Normalize

from sklearn.linear_model import SGDClassifier
import numpy as np
#from NotRunnables import *

class Train():
    dir = ""
    mean = 0
    std = 0
    def __init__(self, dir):
        self.dir = dir

    def train_model(self, hyper_param):
        # load training feature, validation feature
        train_x_file = Path.mean_mfcc_file(self.dir, Path.data[0])
        train_X = np.load(train_x_file)

        # feature normalization
        train_X = train_X.T
        train_X_mean = np.mean(train_X, axis=0)
        self.mean = train_X_mean
        train_X_std = np.std(train_X, axis=0)
        self.std = train_X_std
        train_X = Normalize.normalize(train_X, self.mean, self.std)

        #ToDo: 모든 데이터를 로드할 때는 라벨도 고쳐야함
        # generate labels
        train_Y = Path.train_Y

        # Choose a classifier (here, linear SVM)
        clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param,
                            max_iter=1000, penalty="l2", random_state=0)

        # train
        clf.fit(train_X, train_Y)

        return clf