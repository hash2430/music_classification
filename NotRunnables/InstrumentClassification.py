import FeatureExtraction
import FeatureSummary
import Path
import Test
import Train
import Validate

class InstrumentClassification():
    dir = ""
    def run(self):
        featureExtraction = FeatureExtraction()
        featureExtraction.extract_mfcc1()

        featureSummary = FeatureSummary()
        featureSummary.mean_mfcc()

        train = Train()
        train.svm()

        validate = Validate()
        validate.run()

        test = Test()
        test.run()


