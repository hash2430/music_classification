#import NotRunnables.FeatureExtraction as FeatureExtraction
import NotRunnables.Counter as Counter
from NotRunnables.FeatureExtraction import FeatureExtraction
import NotRunnables.FeatureSummary as FeatureSummary
import NotRunnables.Path as Path
import NotRunnables.Test as Test
import NotRunnables.Train as Train
import NotRunnables.Validate as Validate

class LinearModel():
    mfcc_dir = ""
    mean_mfcc_dir = ""
    report_file = ""

    def __init__(self, mfcc_dir, mean_mfcc_dir, report_file):
        self.mfcc_dir = mfcc_dir
        self.mean_mfcc_dir = mean_mfcc_dir
        self.report_file = report_file

    def run(self):
        # feature extraction and summary
        featureExtraction = FeatureExtraction(self.mfcc_dir)
        featureSummary = FeatureSummary.FeatureSummary(featureExtraction.mfcc_dim, self.mfcc_dir, self.mean_mfcc_dir)

        for phase in range(3):
            featureExtraction.extract_mfcc1(phase)
            featureSummary.mean_mfcc(phase)
            featureSummary.visualize(phase)

        # train
        train = Train.Train(self.mean_mfcc_dir)
        # todo: hyperparam를 하나 이상의 파라미터를 가질 수 있게 확장할 것(조합)
        # iterate among hyper params
        hyper_params = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        models = []
        for hyper_param in hyper_params:
            model = train.linear_svm(hyper_param)
            models.append(model)



        #todo: train_X_mean, train_X_std를 어디 저장할까?
        validate = Validate.Validate(self.mean_mfcc_dir, self.report_file)
        final_model, validation_acc = validate.validate(hyper_params, models,
                                                        train.mean, train.std)
        print("Validation accuracy: " + str(validation_acc))

        test = Test.Test(self.mean_mfcc_dir, self.report_file)
        test_acc = test.test(train.mean, train.std, final_model)
        print("Test accuracy: " + str(test_acc))

class NonLinearSVM():
    pass


