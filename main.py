import os

# if you want to run the code from scratch
os.system("python train_model.py --run 0 17")
os.system("python train_model.py --run 17 34")
os.system("python train_model.py --run 34 51")
os.system("python train_model.py --run 51 68")
os.system("python train_model.py --run 68 84")
os.system("python train_model.py --run 84 100")


# otherwise run the analysis script it will write it to the output folder
os.system("python analysis.py")