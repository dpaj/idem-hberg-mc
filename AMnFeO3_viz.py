from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

E_temperature_array = np.load("1510862710_x=0.0_N=8E_temperature_array.npy")

plt.plot(E_temperature_array.transpose())
plt.show()
