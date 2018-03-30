# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#

from Baseline import util

from Baseline.feature_summary import *

from sklearn.linear_model import SGDClassifier

def train_model(train_X, train_Y, valid_X, valid_Y, hyper_param1):

    # Choose a classifier (here, linear SVM)
    clf = SGDClassifier(verbose=0, loss="hinge", alpha=hyper_param1,
                        max_iter=1000, penalty="l2", random_state=0)

    # train
    clf.fit(train_X, train_Y)

    # validation
    valid_Y_hat = clf.predict(valid_X)#(3,)
    valid_data_size = valid_Y.shape[0]
    accuracy = np.sum((valid_Y_hat == valid_Y))/valid_data_size*100.0
    print('validation accuracy = ' + str(accuracy) + ' %')
    
    return clf, accuracy

if __name__ == '__main__':

    # load data 
    train_X = mean_mfcc('train') #(20,3)=(mfcc_dim, num of files)
    valid_X = mean_mfcc('valid') #(20,3)=(mfcc_dim, num of files)
    test_X = mean_mfcc('test') #(20,3)=(mfcc_dim, num of files)

    # label generation
    # cls = np.array([1,2,3,4,5,6,7,8,9,10])
    # train_Y = np.repeat(cls, 100)
    # valid_Y = np.repeat(cls, 20)
    # test_Y = np.repeat(cls, 20)
    cls = np.array([1, 2, 3])
    train_Y = cls
    valid_Y = cls
    test_Y = cls

    # feature normalizaiton
    train_X = train_X.T #(3, 20)
    train_X_mean = np.mean(train_X, axis=0)#(20,)
    train_X = train_X - train_X_mean#(3,20)
    train_X_std = np.std(train_X, axis=0)#(20,)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T#(3, 20)
    valid_X = valid_X - train_X_mean # validation set을 train set의 분포로 normalize하는 이유는 train set이 더 크기 때문에 분포가 더 정확해서라고 수업에서 말씀하심.
    valid_X = valid_X/(train_X_std + 1e-5)

    # training model
    # validation하여 hyper parameter를 선택하는 validation 과정까지 포함한 시간을 train time으로 쳐보았다
    #util.start_measure("train + validation")
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    model = []
    valid_acc = []
    for a in alphas:
        clf, acc = train_model(train_X, train_Y, valid_X, valid_Y, a)
        model.append(clf)
        valid_acc.append(acc)
    # choose the model that achieve the best validation accuracy
    final_model = model[np.argmax(valid_acc)]
    #util.finish_measure()

    # now, evaluate the model with the test set
    #util.start_measure("test")
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X/(train_X_std + 1e-5)
    test_Y_hat = final_model.predict(test_X)
    test_data_size = test_Y.shape[0]
    accuracy = np.sum((test_Y_hat == test_Y))/test_data_size*100.0
    print('test accuracy = ' + str(accuracy) + ' %')
    #util.finish_measure()

