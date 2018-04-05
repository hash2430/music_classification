# input: mean mfcc files, variance mfcc files
# output: hyperparameters and models files
from NotRunnables import Path, Normalize, Counter

from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import os
#from NotRunnables import *

class Train():
    inputDir = ""
    reportFile = ""
    mean = 0
    std = 0
    def __init__(self, inputDir, reportFile):
        self.inputDir = inputDir
        self.reportFile = reportFile
        if not os.path.exists(os.path.dirname(self.reportFile)):
            os.makedirs(os.path.dirname(self.reportFile))
        file = open(self.reportFile, 'w')
        file.write('********** Train Report **********\n')
        file.write("== Training time ==\n")

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
        counter = Counter.Counter(self.reportFile)
        counter.start_measure("")
        clf.fit(train_X, train_Y)
        counter.finish_measure()


        return clf

    def nonlinear_svm(self, kernel):
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
        clf = svm.NuSVC( nu=0.5, kernel=kernel, degree=3, gamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=None)

        # train
        counter = Counter.Counter(self.reportFile)
        counter.start_measure("")
        clf.fit(train_X, train_Y)
        counter.finish_measure()
        return clf

    def knn(self, k):
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
        clf = KNeighborsClassifier(n_neighbors=k)

        # train
        counter = Counter.Counter(self.reportFile)
        counter.start_measure("")
        clf.fit(train_X, train_Y)
        counter.finish_measure()
        return clf