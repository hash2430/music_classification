import NotRunnables.InstrumentClassification as InstClassf
import NotRunnables.Path as Path
import NotRunnables.Counter as Counter

if __name__ == "__main__":
    # my nickname on leader board is "cs" but i am an ee student.
    # 1. Different learning algorithms
    mfcc_dir = Path.method1_path + "mfcc20/"
    mean_mfcc_dir = Path.method1_path + "mean_mfcc20/"
    report_file = Path.method1_path + "report/mfcc20_linear_svm"

    # 1-1. lenear kernel svm performance: 43.0%
    counter = Counter.Counter(file_name=Path.method1_path + 'time_measure')
    IC1_1 = InstClassf.LinearModel(mfcc_dir, mean_mfcc_dir, report_file)
    counter.start_measure('base line overall start')
    IC1_1.run()
    counter.finish_measure()

    # 1-2. rbf kernerl svm performance: 47.5%
    report_file = Path.method2_path + "report/mfcc20_nonlinear_svm"
    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure20')
    counter.start_measure("mfcc_dim = 20, sr = 22050, learning algorithm = svm with rbf kernel")
    IC1_2 = InstClassf.NonLinearSVM(mfcc_dir, mean_mfcc_dir, report_file)
    IC1_2.run()
    counter.finish_measure()

    # 1-3. knn: 47.0%
    report_file = Path.method2_path + "report/mfcc20_knn"
    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure_knn')
    counter.start_measure("mfcc_dim = 20, sr = 22050, learning algorithm = knn")
    IC1_3 = InstClassf.KNN(mfcc_dir, mean_mfcc_dir, report_file)
    IC1_3.run()
    counter.finish_measure()

    # 2. Different feature summary
    # 2-1. 60 dim mfcc performance: 56.5%
    mfcc_dir = Path.method2_path + "mfcc60/"
    mean_mfcc_dir = Path.method2_path + "mean_mfcc60/"
    report_file = Path.method2_path + "report/mfcc60_nonlinear_svm"

    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure60')
    counter.start_measure("mfcc_dim = 60, sr = 22050, learning algorithm = svm with rbf kernel")
    IC2_1 =  InstClassf.LargerMfccDim(mfcc_dir, mean_mfcc_dir, report_file)
    IC2_1.run()
    counter.finish_measure()

    # 2-2. mfcc var feature concatenated(total 120 features): 63.0%
    mfcc_dir = Path.method2_path + "mfcc60/"
    mean_mfcc_dir = Path.method2_path + "mean_mfcc60/"
    var_mfcc_dir = Path.method2_path + "var_mfcc60/"
    mean_var_mfcc_dir = Path.method2_path + "mean_var_mfcc60/"
    report_file = Path.method2_path + "report/mfcc60_mean_var_nonlinear_svm"

    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure60_mean_var')
    counter.start_measure("mfcc_dim = 60, sr = 22050, learning algorithm = svm with rbf kernel, mfcc var feature added")
    IC2_2 =  InstClassf.MfccVarAdded(mfcc_dir, mean_mfcc_dir, var_mfcc_dir, mean_var_mfcc_dir, report_file)
    IC2_2.run()
    counter.finish_measure()

    # 2-3. mfcc performance: 64%
    mfcc_dir = Path.method2_path + "mfcc60/"
    flat_mfcc_dir = Path.method2_path + "flat_mfcc60"
    report_file = Path.method2_path + "report/mfcc60all"

    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure_all_mfcc')
    counter.start_measure("mfcc_dim = 60, sr = 22050, learning algorithm = svm with rbf kernel, all mfcc")
    IC2_3 =  InstClassf.MfccAll(mfcc_dir, flat_mfcc_dir, report_file)
    IC2_3.run()
    counter.finish_measure()


    # 2-4. mfcc delta (1st and 2nd order) feature concatenated(total 180 features): 65.5%
    mfcc_dir = Path.method2_path + "mfcc60/"
    mean_var_mfcc_dir = Path.method2_path + "mean_var_mfcc60/"
    delta_mfcc_file = Path.method2_path + "delta_mfcc60/"
    mean_var_delta_mfcc_file = Path.method2_path + "mean_var_delta_mfcc60/"
    report_file = Path.method2_path + "report/report60_mean_var_delta"

    counter = Counter.Counter(file_name=Path.method2_path + 'time_measure60_mean_var')
    counter.start_measure("mfcc_dim = 60, sr = 22050, learning algorithm = svm with rbf kernel, mfcc var, delta feature added.\n total 180 dim features")
    IC2_4 =  InstClassf.MfccDeltaAdded(mfcc_dir, mean_var_mfcc_dir, delta_mfcc_file, mean_var_delta_mfcc_file, report_file)
    IC2_4.run()
    counter.finish_measure()

    # 3. feature extraction
    # 3-1. stft: 60.5
    mfcc_dir = Path.method3_path + "mfcc60_stft/"
    mean_mfcc_dir = Path.method3_path + "mfcc60_mean_stft"
    var_mfcc_dir = Path.method3_path + "mfcc60_var_stft"
    mean_var_mfcc_dir = Path.method3_path + "mean_var_mfcc60_stft/"
    report_file = Path.method3_path + "report/mfcc60_mean_var_stft"

    counter = Counter.Counter(file_name=Path.method3_path + 'time_measure60_mean_var_stft')
    counter.start_measure("mfcc_dim = 60, sr = 22050, learning algorithm = svm with rbf kernel, mfcc var. stft adjusted")
    IC3_1 =  InstClassf.FEStft(mfcc_dir, mean_mfcc_dir, var_mfcc_dir, mean_var_mfcc_dir, report_file)
    IC3_1.run()
    counter.finish_measure()

    print("-Fin.-")