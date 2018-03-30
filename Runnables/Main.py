from NotRunnables import *
import NotRunnables.InstrumentClassification as InstClassf
if __name__ == "__main__":
    dir1 = Path.method1_path
    IC1 = InstClassf.LinearModel(dir1)
    IC1.run()

    dir2 = Path.method2_path
    IC2 = InstClassf.NonLinearSVM(dir2)
    IC2.run()

    print("-Fin.-")