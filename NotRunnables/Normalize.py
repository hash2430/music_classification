#from NotRunnables import *

def normalize(data, mean, std):
    data = data - mean
    data = data / (std + 1e-5)
    return data
