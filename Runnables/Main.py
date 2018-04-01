import NotRunnables.InstrumentClassification as InstClassf
import NotRunnables.Path as Path
import NotRunnables.Counter as Counter
if __name__ == "__main__":
    mfcc_dir = Path.method1_path + "mfcc11/"
    mean_mfcc_dir = Path.method1_path + "mean_mfcc11/"
    report_file = Path.method1_path + "report11"

    counter = Counter.Counter()
    dir1 = Path.method1_path
    IC1 = InstClassf.LinearModel(mfcc_dir, mean_mfcc_dir, report_file)
    counter.start_measure('base line overall start')
    IC1.run()
    counter.finish_measure()
    # dir2 = Path.method2_path
    # IC2 = InstClassf.NonLinearSVM(dir2)
    # IC2.run()

    print("-Fin.-")