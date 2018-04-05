#import NotRunnables.FeatureExtraction as FeatureExtraction
import NotRunnables.Counter as Counter
from NotRunnables.FeatureExtraction import FeatureExtraction
import NotRunnables.FeatureSummary as FeatureSummary
import NotRunnables.Path as Path
import NotRunnables.Test as Test
import NotRunnables.Train as Train
import NotRunnables.Validate as Validate
import numpy as np
class Classification():
    mfcc_dir = ""
    input1_dir = ""
    report_file = ""

    def __init__(self, mfcc_dir, mean_mfcc_dir, report_file):
        self.mfcc_dir = mfcc_dir
        self.input1_dir = mean_mfcc_dir
        self.report_file = report_file

    def run(self):
        pass

class LinearModel(Classification):
    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir)
        featureSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.input1_dir)

        for phase in range(3):
            featureExtraction.extract_mfcc1(phase)
            featureSummary.mean_mfcc(phase)
            featureSummary.visualize(phase)

        # train
        train = Train.Train(Path.mean_mfcc_file(self.input1_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        models = []
        for hyper_param in hyper_params:
            model = train.linear_svm(hyper_param)
            models.append(model)



        #todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_mfcc_file(self.input1_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_mfcc_file(self.input1_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class NonLinearSVM(Classification):
    def run(self):
        # train
        train = Train.Train(Path.mean_mfcc_file(self.input1_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)



        #todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_mfcc_file(self.input1_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_mfcc_file(self.input1_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class KNN(Classification):
    def run(self):
        # train
        train = Train.Train(Path.mean_mfcc_file(self.input1_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = [3, 5, 7]
        models = []
        for hyper_param in hyper_params:
            model = train.knn(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_mfcc_file(self.input1_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_mfcc_file(self.input1_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))






class LargerMfccDim(Classification):
    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir,mfcc_dim=60)
        featureSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.input1_dir)

        for phase in range(3):
            featureExtraction.extract_mfcc1(phase)
            featureSummary.mean_mfcc(phase)
            featureSummary.visualize(phase)

        # train
        train = Train.Train(Path.mean_mfcc_file(self.input1_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_mfcc_file(self.input1_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_mfcc_file(self.input1_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class MfccVarAdded(Classification):
    input2_dir = ""
    concat_dir = ""

    def __init__(self, mfcc_dir, mean_mfcc_dir, var_mfcc_dir, concat_dir, report_file):
        self.mfcc_dir = mfcc_dir
        self.input1_dir = mean_mfcc_dir
        self.input2_dir = var_mfcc_dir
        self.concat_dir = concat_dir
        self.report_file = report_file

    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir, mfcc_dim=60)
        featureVarSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.input2_dir)
        concatenator = FeatureSummary.FeatureConcatenator()
        for phase in range(3):
            featureVarSummary.var_mfcc(phase)
            step = Path.data[phase]
            concatenator.concat(Path.mean_mfcc_file(self.input1_dir, step),
                                Path.var_mfcc_file(self.input2_dir, step),
                                Path.mean_var_mfcc_file(self.concat_dir, step))

        # train
        train = Train.Train(Path.mean_var_mfcc_file(self.concat_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_var_mfcc_file(self.concat_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_var_mfcc_file(self.concat_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class MfccDeltaAdded(MfccVarAdded):
    def run(self):
        # feature extraction and summary
        featureDelta = FeatureSummary.FeatureSummary(60, self.mfcc_dir, self.input2_dir)
        concatenator = FeatureSummary.FeatureConcatenator()
        for phase in range(3):
            featureDelta.delta_mfcc(phase)
            step = Path.data[phase]
            concatenator.concat(Path.mean_var_mfcc_file(self.input1_dir, step),
                                Path.delta_mfcc_file(self.input2_dir, step),
                                Path.mean_var_delta_mean_mfcc_file(self.concat_dir, step))

        # train
        train = Train.Train(Path.mean_var_delta_mean_mfcc_file(self.concat_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_var_delta_mean_mfcc_file(self.concat_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_var_delta_mean_mfcc_file(self.concat_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class MfccAll(Classification):
    flat_mfcc_dir = ""

    def __init__(self, mfcc_dir, flat_mfcc_dir, report_file):
        self.mfcc_dir = mfcc_dir
        self.flat_mfcc_dir = flat_mfcc_dir
        self.report_file = report_file

    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir, mfcc_dim=60)
        featureSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.flat_mfcc_dir)

        for phase in range(3):
            featureSummary.flatten(phase)

        # train
        train = Train.Train(Path.flat_mfcc_file(self.flat_mfcc_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.flat_mfcc_file(self.flat_mfcc_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.flat_mfcc_file(self.flat_mfcc_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class FEStft(MfccVarAdded):
    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir, mfcc_dim=60)
        featureMeanSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir,
                                                           self.input1_dir)
        featureVarSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.input2_dir)
        concatenator = FeatureSummary.FeatureConcatenator()
        for phase in range(3):
            featureExtraction.extract_mfcc2(phase)
            featureMeanSummary.mean_mfcc(phase)
            featureMeanSummary.visualize(phase)
            featureVarSummary.var_mfcc(phase)
            step = Path.data[phase]
            concatenator.concat(Path.mean_mfcc_file(self.input1_dir, step),
                                Path.var_mfcc_file(self.input2_dir, step),
                                Path.mean_var_mfcc_file(self.concat_dir, step))

        # train
        train = Train.Train(Path.mean_var_mfcc_file(self.concat_dir, Path.data[0]), self.report_file)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = ['rbf', 'poly', 'sigmoid']
        models = []
        for hyper_param in hyper_params:
            model = train.nonlinear_svm(hyper_param)
            models.append(model)

        # todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(Path.mean_var_mfcc_file(self.concat_dir, Path.data[1]), self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(Path.mean_var_mfcc_file(self.concat_dir, Path.data[2]), self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))
