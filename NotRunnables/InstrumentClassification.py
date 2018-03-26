import FeatureExtraction
import FeatureSummary
import Path
import Test
import Train
import Validate

class InstrumentClassification():
    dir = ""
    def __init__(self, dir):
        self.dir = dir

    def run(self):
        featureExtraction = FeatureExtraction(self.dir)
        featureExtraction.extract_mfcc1()

        featureSummary = FeatureSummary(self.dir)
        featureSummary.mean_mfcc()
        featureSummary.visualize()

        train = Train(self.dir)
        train.svm()

        validate = Validate(self.dir)
        validate.run()

        test = Test(self.dir)
        test.run()


