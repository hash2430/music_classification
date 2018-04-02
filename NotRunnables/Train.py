# input: mean mfcc files, variance mfcc files
# output: hyperparameters and models files
from NotRunnables import Path, Normalize

from sklearn.linear_model import SGDClassifier
from sklearn import svm

import numpy as np
#from NotRunnables import *

class Train():
    inputDir = ""
    mean = 0
    std = 0
    def __init__(self, inputDir):
        self.inputDir = inputDir

    def linear_svm(self, hyper_param):
        # load training feature, validation feature
        train_x_file = self.inputDir
        train_X = np.load(train_x_file)

        # feature normalization: feature summary 저장 단계가 아니라
        #  train에서 normalize가 일어남으로써 확실하게 feature normalization을 잊지 않고 할 수 있다.
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

    def nonlinear_svm(self):
        train_x_file = self.inputDir
        train_X = np.load(train_x_file)

        # feature normalization
        train_X = train_X.T
        train_X_mean = np.mean(train_X, axis=0)
        self.mean = train_X_mean
        train_X_std = np.std(train_X, axis=0)
        self.std = train_X_std
        train_X = Normalize.normalize(train_X, self.mean, self.std)

        # ToDo: 모든 데이터를 로드할 때는 라벨도 고쳐야함
        # generate labels
        train_Y = Path.train_Y

        # Choose a classifier (here, linear SVM)
        clf = svm.NuSVC()

        # train
        clf.fit(train_X, train_Y)
        return clf