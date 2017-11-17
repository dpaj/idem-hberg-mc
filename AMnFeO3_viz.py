from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

file_time = "1510867146"

x = "0.0"

N = "24"

file_prefix = file_time+"_x="+x+"_N="+N

steps_to_burn = 0

temperature_sweep_array = np.load(file_prefix+"_temperature_sweep_array.npy")

E_temperature_array = np.load(file_prefix+"_E_temperature_array.npy")
E_mean_temperature_array = np.zeros(np.shape(temperature_sweep_array))
E_std_temperature_array = np.zeros(np.shape(temperature_sweep_array))

A_temperature_array = np.load(file_prefix+"_A_temperature_array.npy")
A_mean_temperature_array = np.zeros((3,3,len(temperature_sweep_array)))
A_std_temperature_array = np.zeros((3,3,len(temperature_sweep_array)))

G_temperature_array = np.load(file_prefix+"_G_temperature_array.npy")
G_mean_temperature_array = np.zeros((3,3,len(temperature_sweep_array)))
G_std_temperature_array = np.zeros((3,3,len(temperature_sweep_array)))

nn_pair_corr_b_temperature_array = np.load(file_prefix+"_nn_pair_corr_b_temperature_array.npy")
nn_pair_corr_b_mean_temperature_array = np.zeros(np.shape(temperature_sweep_array))
nn_pair_corr_b_std_temperature_array = np.zeros(np.shape(temperature_sweep_array))

nn_pair_corr_ac_temperature_array = np.load(file_prefix+"_nn_pair_corr_ac_temperature_array.npy")
nn_pair_corr_ac_mean_temperature_array = np.zeros(np.shape(temperature_sweep_array))
nn_pair_corr_ac_std_temperature_array = np.zeros(np.shape(temperature_sweep_array))



print(temperature_sweep_array)

for temperature_index, temperature in enumerate(temperature_sweep_array):
	for equilibration_index in range(steps_to_burn+1, np.shape(E_temperature_array)[1]):
		E_mean_temperature_array[temperature_index] = np.mean(E_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		E_std_temperature_array[temperature_index] = np.std(E_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		nn_pair_corr_b_mean_temperature_array[temperature_index] = np.mean(nn_pair_corr_b_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		nn_pair_corr_b_std_temperature_array[temperature_index] = np.std(nn_pair_corr_b_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		nn_pair_corr_ac_mean_temperature_array[temperature_index] = np.mean(nn_pair_corr_ac_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		nn_pair_corr_ac_std_temperature_array[temperature_index] = np.std(nn_pair_corr_ac_temperature_array[temperature_index, steps_to_burn:equilibration_index])




for atom_index in range(3):
	for cartesian_direction_index in range(3):
		for temperature_index, temperature in enumerate(temperature_sweep_array):
			for equilibration_index in range(steps_to_burn+1, np.shape(E_temperature_array)[1]):
				A_mean_temperature_array[atom_index,cartesian_direction_index,temperature_index] = np.mean(A_temperature_array[atom_index,cartesian_direction_index,temperature_index,steps_to_burn:equilibration_index])
				A_std_temperature_array[atom_index,cartesian_direction_index,temperature_index] = np.std(A_temperature_array[atom_index,cartesian_direction_index,temperature_index,steps_to_burn:equilibration_index])
				G_mean_temperature_array[atom_index,cartesian_direction_index,temperature_index] = np.mean(G_temperature_array[atom_index,cartesian_direction_index,temperature_index,steps_to_burn:equilibration_index])
				G_std_temperature_array[atom_index,cartesian_direction_index,temperature_index] = np.std(G_temperature_array[atom_index,cartesian_direction_index,temperature_index,steps_to_burn:equilibration_index])

				

plt.plot(E_temperature_array.transpose())
plt.figure()
plt.plot(temperature_sweep_array, E_mean_temperature_array)
plt.plot(temperature_sweep_array, E_std_temperature_array)
plt.plot(temperature_sweep_array, np.gradient(E_mean_temperature_array))

plt.figure()
plt.plot(temperature_sweep_array, nn_pair_corr_b_mean_temperature_array)
plt.plot(temperature_sweep_array, nn_pair_corr_b_std_temperature_array)
plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_b_mean_temperature_array))
plt.plot(temperature_sweep_array, nn_pair_corr_ac_mean_temperature_array)
plt.plot(temperature_sweep_array, nn_pair_corr_ac_std_temperature_array)
plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_ac_mean_temperature_array))

plt.figure()
plt.plot(temperature_sweep_array, A_mean_temperature_array[0,0,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[0,1,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[0,2,:])

plt.plot(temperature_sweep_array, A_mean_temperature_array[1,0,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[1,1,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[1,2,:])

plt.plot(temperature_sweep_array, A_mean_temperature_array[2,0,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[2,1,:])
plt.plot(temperature_sweep_array, A_mean_temperature_array[2,2,:])

plt.figure()
plt.plot(temperature_sweep_array, A_std_temperature_array[0,0,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[0,1,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[0,2,:])

plt.plot(temperature_sweep_array, A_std_temperature_array[1,0,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[1,1,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[1,2,:])

plt.plot(temperature_sweep_array, A_std_temperature_array[2,0,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[2,1,:])
plt.plot(temperature_sweep_array, A_std_temperature_array[2,2,:])
		
plt.figure()
plt.plot(temperature_sweep_array, G_mean_temperature_array[0,0,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[0,1,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[0,2,:])

plt.plot(temperature_sweep_array, G_mean_temperature_array[1,0,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[1,1,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[1,2,:])

plt.plot(temperature_sweep_array, G_mean_temperature_array[2,0,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[2,1,:])
plt.plot(temperature_sweep_array, G_mean_temperature_array[2,2,:])

plt.figure()
plt.plot(temperature_sweep_array, G_std_temperature_array[0,0,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[0,1,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[0,2,:])

plt.plot(temperature_sweep_array, G_std_temperature_array[1,0,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[1,1,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[1,2,:])

plt.plot(temperature_sweep_array, G_std_temperature_array[2,0,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[2,1,:])
plt.plot(temperature_sweep_array, G_std_temperature_array[2,2,:])

		
		
plt.show()
