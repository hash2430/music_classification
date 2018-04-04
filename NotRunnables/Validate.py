# input: models, validation set
# output: performance sheet, model

#from NotRunnables import *
from NotRunnables import Path, Normalize

import numpy as np
import os

class Validate():
    mean_mfcc_dir = ""
    refportFile = ""

    def __init__(self, mean_mfcc_dir, reportFile):
        self.mean_mfcc_dir = mean_mfcc_dir
        self.reportFile = reportFile

    # hyper_params list is for reporting purpose only
    # since clfs are already calculated in 'Train' class
    def validate(self, hyper_params, clfs, train_X_mean, train_X_std):

        # load validation set
        valid_file = self.mean_mfcc_dir
        valid_X = np.load(valid_file)

        # feature normalization
        valid_X = valid_X.T
        valid_X = Normalize.normalize(valid_X, train_X_mean, train_X_std)

        # label generation
        valid_Y = Path.valid_Y

        # validation
        accuracy_list = []
        for clf in clfs:
            valid_Y_hat = clf.predict(valid_X)
            valid_data_size = valid_Y.shape[0]
            acc = np.sum((valid_Y_hat == valid_Y)) / valid_data_size * 100.0
            accuracy_list.append(acc)
        best_model_idx = np.argmax(accuracy_list)
        final_model = clfs[best_model_idx]

        # ToDo: report file formatting
        # report validation result as file output
        save_file = self.reportFile
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        report_file = open(save_file, 'a')
        report_file.write('\n********** Validation Report **********\n')
        report_file.write('== hyper parameters ==\n')
        for param in hyper_params:
            report_file.write(str(param) + "\n")

        report_file.write('\n')
        # report_file.write('== models ==\n')
        # report_file.write(str(clfs))
        report_file.write('== accuracy ==\n')
        for acc in accuracy_list:
            report_file.write('%8.6f'%(acc) + '%' + '\n')
        report_file.write('== best model ==\n')
        report_file.write(str(final_model) + '\n\n')
        report_file.write('Validation Accuracy: ' + str(accuracy_list[best_model_idx]) + '%' + '\n')

        return final_model, accuracy_list[best_model_idx]



