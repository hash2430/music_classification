import NotRunnables.InstrumentClassification as InstClassf
import NotRunnables.Path as Path
if __name__ == "__main__":
    mfcc_dir = Path.method1_path + "mfcc11/"
    mean_mfcc_dir = Path.method1_path + "mean_mfcc11/"
    report_file = Path.method1_path + "report11"

    dir1 = Path.method1_path
    IC1 = InstClassf.LinearModel(mfcc_dir, mean_mfcc_dir, report_file)
    IC1.run()

    # dir2 = Path.method2_path
    # IC2 = InstClassf.NonLinearSVM(dir2)
    # IC2.run()

    print("-Fin.-")