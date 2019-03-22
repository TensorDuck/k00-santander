#load all data:
import numpy as np

q = np.loadtxt("train.csv", delimiter=",", skiprows=1, usecols=np.arange(1,202))
np.save("train_inputs.npy", q[:,1:])
np.save("train_targets.npy", q[:,0])

w = np.loadtxt("test.csv", delimiter=",", skiprows=1, usecols=np.arange(1,201))
np.save("test_inputs.npy", w)
