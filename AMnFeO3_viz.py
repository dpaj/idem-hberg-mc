from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as spop

#file_time = "1510970159" #x =0.0, L = 4
#file_time = "1510970392" #x =1.0, L = 4
#file_time = "1510970584" #x = 0.2, L = 4
#file_time = "1510973615" #x=0.2, L = 8
file_time = "1510974914" #x=0.2, L = 8

x = "0.0"

L = "8"

file_prefix = file_time+"_x="+x+"_L="+L

x = float(x)

steps_to_burn = 10

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

nn_pair_corr_abs_abc_temperature_array = np.load(file_prefix+"_nn_pair_corr_abs_abc_temperature_array.npy")
nn_pair_corr_abs_abc_mean_temperature_array = np.zeros(np.shape(temperature_sweep_array))
nn_pair_corr_abs_abc_std_temperature_array = np.zeros(np.shape(temperature_sweep_array))

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
		nn_pair_corr_abs_abc_mean_temperature_array[temperature_index] = np.mean(nn_pair_corr_abs_abc_temperature_array[temperature_index, steps_to_burn:equilibration_index])
		nn_pair_corr_abs_abc_std_temperature_array[temperature_index] = np.std(nn_pair_corr_abs_abc_temperature_array[temperature_index, steps_to_burn:equilibration_index])
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

				

f, axarr = plt.subplots(3, 3, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
				
axarr[0, 0].plot(E_temperature_array.transpose())
axarr[0, 0].set_title('energy equilibrations')

axarr[0, 1].plot(temperature_sweep_array, E_mean_temperature_array)
axarr[0, 1].plot(temperature_sweep_array, E_std_temperature_array)
axarr[0, 1].plot(temperature_sweep_array, np.gradient(E_mean_temperature_array))

axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_abs_abc_mean_temperature_array,label='|a|+|b|+|c|')
#plt.plot(temperature_sweep_array, nn_pair_corr_abs_abc_std_temperature_array)
axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_b_mean_temperature_array, label = 'b')
#plt.plot(temperature_sweep_array, nn_pair_corr_b_std_temperature_array)
#plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_b_mean_temperature_array))
axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_ac_mean_temperature_array, label = 'a+c')
#plt.plot(temperature_sweep_array, nn_pair_corr_ac_std_temperature_array)
#plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_ac_mean_temperature_array))
axarr[0, 0].set_title('n.n. pair correlation')
axarr[0, 2].legend()


axarr[1, 0].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[0,0,:]**2+A_mean_temperature_array[0,1,:]**2+A_mean_temperature_array[0,2,:]**2),'ko-',label='a$_{tot}$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,0,:],'o-',label='a$_x$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,1,:],'o-',label='a$_y$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,2,:],'o-',label='a$_z$')

A_fit_temperature_min = 30
A_fit_temperature_max = 65
A_tot_mean_temperature_array = np.sqrt(A_mean_temperature_array[0,0,:]**2+A_mean_temperature_array[0,1,:]**2+A_mean_temperature_array[0,2,:]**2)



print(temperature_sweep_array[ (temperature_sweep_array>=A_fit_temperature_min) & (temperature_sweep_array<=A_fit_temperature_max) ])
print(A_tot_mean_temperature_array[ (temperature_sweep_array>=A_fit_temperature_min) & (temperature_sweep_array<=A_fit_temperature_max) ])

def a_op_fit(x, super_exchange_field, single_ion_anisotropy_ijk, s_max_ijk):
	"(T-T_N)^beta"


"""
spop.optimize.fmin(a_op_fit, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5,\
					x0=(theta[i,j,k], phi[i,j,k]), args = (super_exchange_field_c,single_ion_anisotropy_ijk,s_max_ijk), disp=0)
"""


axarr[1, 0].legend()

axarr[1, 1].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[1,0,:]**2+A_mean_temperature_array[1,1,:]**2+A_mean_temperature_array[1,2,:]**2)/(1-x),'ko-',label='mn_a$_{tot}$')
axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,0,:]/(1-x),label='mn_a$_x$')
axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,1,:]/(1-x),label='mn_a$_y$')
axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,2,:]/(1-x),label='mn_a$_z$')
axarr[1, 1].legend()

axarr[1, 2].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[2,0,:]**2+A_mean_temperature_array[2,1,:]**2+A_mean_temperature_array[2,2,:]**2)/x,'ko-',label='fe_a$_{tot}$')
axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,0,:]/x,label='fe_a$_x$')
axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,1,:]/x,label='fe_a$_y$')
axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,2,:]/x,label='fe_a$_z$')
axarr[1, 2].legend()

axarr[2, 0].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[0,0,:]**2+G_mean_temperature_array[0,1,:]**2+G_mean_temperature_array[0,2,:]**2),'ko-',label='g$_{tot}$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,0,:],'.-',label='g$_x$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,1,:],'.-',label='g$_y$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,2,:],'.-',label='g$_z$')
axarr[2, 0].legend()

axarr[2, 1].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[1,0,:]**2+G_mean_temperature_array[1,1,:]**2+G_mean_temperature_array[1,2,:]**2)/(1-x),'ko-',label='mn_g$_{tot}$')
axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,0,:]/(1-x),label='mn_g$_x$')
axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,1,:]/(1-x),label='mn_g$_y$')
axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,2,:]/(1-x),label='mn_g$_z$')
axarr[2, 1].legend()

axarr[2, 2].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[2,0,:]**2+G_mean_temperature_array[2,1,:]**2+G_mean_temperature_array[2,2,:]**2)/x,'ko-',label='fe_g$_{tot}$')
axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,0,:]/x,label='fe_g$_x$')
axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,1,:]/x,label='fe_g$_y$')
axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,2,:]/x,label='fe_g$_z$')
axarr[2, 2].legend()
"""
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

		
"""		
plt.suptitle('edge_length='+str(L)+', iron_doping_level='+str(x))
plt.show()
