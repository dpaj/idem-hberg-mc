from __future__ import print_function
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as spop

def op_fit(x, temperature_sweep_array_slice, A_tot_mean_temperature_array_slice):
	#fit an arbitrary scale factor, T_N, and beta
	arbitrary_scale_factor = x[0]
	T_N = x[1]
	beta = x[2]
	residuals = np.sum(( A_tot_mean_temperature_array_slice - np.multiply(arbitrary_scale_factor,np.power((np.divide(-temperature_sweep_array_slice+T_N, T_N)),beta)) )**2)
	return residuals

file_time = "1511284876" #x=0.0, L=4
file_time = "1511285366" #x=0.0, L=8
file_time = "1511286842" #x=0.0, L=8, D=[-4,0,0]
file_time = "1511289179"
file_time = "1511289327"
file_time = "1511285366"

x = "0.0"

L = "8"

file_prefix = "J1S1_"+file_time+"_x="+x+"_L="+L

x = float(x)
edge_length = int(L)

steps_to_burn = 25

A_fit_temperature_min = 1.3
A_fit_temperature_max = 1.6
G_fit_temperature_min = 1.3
G_fit_temperature_max = 1.6

s_x = np.load(file_prefix+"_s_x.npy")
s_y = np.load(file_prefix+"_s_y.npy")
s_z = np.load(file_prefix+"_s_z.npy")

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

		
### do some intermediary calculations before displaying
A_tot_mean_temperature_array = np.sqrt(A_mean_temperature_array[0,0,:]**2+A_mean_temperature_array[0,1,:]**2+A_mean_temperature_array[0,2,:]**2)
G_tot_mean_temperature_array = np.sqrt(G_mean_temperature_array[0,0,:]**2+G_mean_temperature_array[0,1,:]**2+G_mean_temperature_array[0,2,:]**2)


A_temperature_sweep_array_slice = temperature_sweep_array[ (temperature_sweep_array>=A_fit_temperature_min) & (temperature_sweep_array<=A_fit_temperature_max) ]
A_tot_mean_temperature_array_slice = A_tot_mean_temperature_array[ (temperature_sweep_array>=A_fit_temperature_min) & (temperature_sweep_array<=A_fit_temperature_max) ]


G_temperature_sweep_array_slice = temperature_sweep_array[ (temperature_sweep_array>=G_fit_temperature_min) & (temperature_sweep_array<=G_fit_temperature_max) ]
G_tot_mean_temperature_array_slice = G_tot_mean_temperature_array[ (temperature_sweep_array>=G_fit_temperature_min) & (temperature_sweep_array<=G_fit_temperature_max) ]

a_xfit = spop.optimize.fmin(op_fit, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5,\
					x0=(1.0, 1.1, 0.333), args = (A_temperature_sweep_array_slice, A_tot_mean_temperature_array_slice), disp=1)

a_arbitrary_scale_factor = a_xfit[0]
a_T_N = a_xfit[1]
a_beta = a_xfit[2]
A_temperature_sweep_array_slice_manypoints = np.linspace(A_fit_temperature_min, A_fit_temperature_max, 100)

xfit = spop.optimize.fmin(op_fit, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5,\
					x0=(1.0, 1.8, 0.333), args = (G_temperature_sweep_array_slice, G_tot_mean_temperature_array_slice), disp=1)

g_arbitrary_scale_factor = xfit[0]
g_T_N = xfit[1]
g_beta = xfit[2]
g_temperature_sweep_array_slice_manypoints = np.linspace(G_fit_temperature_min, G_fit_temperature_max, 100)


		

f, axarr = plt.subplots(3, 3, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
				
axarr[0, 0].plot(E_temperature_array.transpose())
axarr[0, 0].set_title('energy equilibrations')

#axarr[0, 1].plot(temperature_sweep_array, E_mean_temperature_array)
axarr[0, 1].plot(temperature_sweep_array, E_std_temperature_array**2/temperature_sweep_array**2)
#axarr[0, 1].plot(temperature_sweep_array, np.gradient(E_mean_temperature_array))

axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_abs_abc_mean_temperature_array,label='|a|+|b|+|c|')
#plt.plot(temperature_sweep_array, nn_pair_corr_abs_abc_std_temperature_array)
axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_b_mean_temperature_array, label = 'b')
#plt.plot(temperature_sweep_array, nn_pair_corr_b_std_temperature_array)
#plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_b_mean_temperature_array))
axarr[0, 2].plot(temperature_sweep_array, nn_pair_corr_ac_mean_temperature_array, label = 'a+c')
#plt.plot(temperature_sweep_array, nn_pair_corr_ac_std_temperature_array)
#plt.plot(temperature_sweep_array, np.gradient(nn_pair_corr_ac_mean_temperature_array))
axarr[0, 2].text(np.max(temperature_sweep_array)/2.0, np.max(nn_pair_corr_abs_abc_mean_temperature_array)*0.8, str("T$_{min}$ collinearity="+str(nn_pair_corr_abs_abc_mean_temperature_array[-1]/3.0)), fontsize=12)
axarr[0, 0].set_title('n.n. pair correlation')
axarr[0, 2].legend()


axarr[1, 0].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[0,0,:]**2+A_mean_temperature_array[0,1,:]**2+A_mean_temperature_array[0,2,:]**2),'ko-',label='a$_{tot}$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,0,:],'o-',label='a$_x$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,1,:],'o-',label='a$_y$')
axarr[1, 0].plot(temperature_sweep_array, A_mean_temperature_array[0,2,:],'o-',label='a$_z$')


axarr[1, 0].plot(A_temperature_sweep_array_slice_manypoints,np.multiply(a_arbitrary_scale_factor,np.power((np.divide(-A_temperature_sweep_array_slice_manypoints+a_T_N, a_T_N)),a_beta)))
axarr[1, 0].text(a_T_N, a_xfit[0]/2.0, str(str(a_xfit[0])+"\n"+str(a_xfit[1])+"\n"+str(a_xfit[2])), fontsize=12)
axarr[1, 0].legend()

if x != 1:
	axarr[1, 1].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[1,0,:]**2+A_mean_temperature_array[1,1,:]**2+A_mean_temperature_array[1,2,:]**2)/(1-x),'ko-',label='mn_a$_{tot}$')

	axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,0,:]/(1-x),label='mn_a$_x$')
	axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,1,:]/(1-x),label='mn_a$_y$')
	axarr[1, 1].plot(temperature_sweep_array, A_mean_temperature_array[1,2,:]/(1-x),label='mn_a$_z$')
	axarr[1, 1].text(np.max(temperature_sweep_array)/2.0, (np.sqrt(A_mean_temperature_array[1,0,:]**2+A_mean_temperature_array[1,1,:]**2+A_mean_temperature_array[1,2,:]**2)/(1-x))[-1]*0.8, str((np.sqrt(A_mean_temperature_array[1,0,:]**2+A_mean_temperature_array[1,1,:]**2+A_mean_temperature_array[1,2,:]**2)/(1-x))[-1]), fontsize=12)
	
	axarr[1, 1].legend()

if x != 0:
	axarr[1, 2].plot(temperature_sweep_array, np.sqrt(A_mean_temperature_array[2,0,:]**2+A_mean_temperature_array[2,1,:]**2+A_mean_temperature_array[2,2,:]**2)/x,'ko-',label='fe_a$_{tot}$')
	axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,0,:]/x,label='fe_a$_x$')
	axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,1,:]/x,label='fe_a$_y$')
	axarr[1, 2].plot(temperature_sweep_array, A_mean_temperature_array[2,2,:]/x,label='fe_a$_z$')
	axarr[1, 2].text(np.max(temperature_sweep_array)/2.0, (np.sqrt(A_mean_temperature_array[2,0,:]**2+A_mean_temperature_array[2,1,:]**2+A_mean_temperature_array[2,2,:]**2)/x)[-1]*0.8, str((np.sqrt(A_mean_temperature_array[2,0,:]**2+A_mean_temperature_array[2,1,:]**2+A_mean_temperature_array[2,2,:]**2)/x)[-1]), fontsize=12)

	axarr[1, 2].legend()

axarr[2, 0].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[0,0,:]**2+G_mean_temperature_array[0,1,:]**2+G_mean_temperature_array[0,2,:]**2),'ko-',label='g$_{tot}$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,0,:],'.-',label='g$_x$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,1,:],'.-',label='g$_y$')
axarr[2, 0].plot(temperature_sweep_array, G_mean_temperature_array[0,2,:],'.-',label='g$_z$')
axarr[2, 0].plot(g_temperature_sweep_array_slice_manypoints,np.multiply(g_arbitrary_scale_factor,np.power((np.divide(-g_temperature_sweep_array_slice_manypoints+g_T_N, g_T_N)),g_beta)))
axarr[2, 0].text(g_T_N, xfit[0]/2.0, str(str(xfit[0])+"\n"+str(xfit[1])+"\n"+str(xfit[2])), fontsize=12)

axarr[2, 0].legend()

if x != 1.0:
	axarr[2, 1].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[1,0,:]**2+G_mean_temperature_array[1,1,:]**2+G_mean_temperature_array[1,2,:]**2)/(1-x),'ko-',label='mn_g$_{tot}$')
	axarr[2, 1].text(np.max(temperature_sweep_array)/2.0, (np.sqrt(G_mean_temperature_array[1,0,:]**2+G_mean_temperature_array[1,1,:]**2+G_mean_temperature_array[1,2,:]**2)/(1-x))[-1]*0.8, str((np.sqrt(G_mean_temperature_array[1,0,:]**2+G_mean_temperature_array[1,1,:]**2+G_mean_temperature_array[1,2,:]**2)/(1-x))[-1]), fontsize=12)
	axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,0,:]/(1-x),label='mn_g$_x$')
	axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,1,:]/(1-x),label='mn_g$_y$')
	axarr[2, 1].plot(temperature_sweep_array, G_mean_temperature_array[1,2,:]/(1-x),label='mn_g$_z$')
	axarr[2, 1].legend()

if x != 0:
	axarr[2, 2].plot(temperature_sweep_array, np.sqrt(G_mean_temperature_array[2,0,:]**2+G_mean_temperature_array[2,1,:]**2+G_mean_temperature_array[2,2,:]**2)/x,'ko-',label='fe_g$_{tot}$')
	axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,0,:]/x,label='fe_g$_x$')
	axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,1,:]/x,label='fe_g$_y$')
	axarr[2, 2].plot(temperature_sweep_array, G_mean_temperature_array[2,2,:]/x,label='fe_g$_z$')
	axarr[2, 2].text(np.max(temperature_sweep_array)/2.0, (np.sqrt(G_mean_temperature_array[2,0,:]**2+G_mean_temperature_array[2,1,:]**2+G_mean_temperature_array[2,2,:]**2)/x)[-1]*0.8, str((np.sqrt(G_mean_temperature_array[2,0,:]**2+G_mean_temperature_array[2,1,:]**2+G_mean_temperature_array[2,2,:]**2)/x)[-1]), fontsize=12)
	axarr[2, 2].legend()
else:
	axarr[2, 2].loglog(np.divide(-g_temperature_sweep_array_slice_manypoints+g_T_N, g_T_N),np.multiply(g_arbitrary_scale_factor,np.power((np.divide(-g_temperature_sweep_array_slice_manypoints+g_T_N, g_T_N)),g_beta)))
	axarr[2, 2].loglog(np.divide(-temperature_sweep_array+g_T_N, g_T_N), np.sqrt(G_mean_temperature_array[0,0,:]**2+G_mean_temperature_array[0,1,:]**2+G_mean_temperature_array[0,2,:]**2),'ko',label='g$_{tot}$')

plt.suptitle('file_prefix='+str(file_prefix+file_time)+'edge_length='+str(L)+', iron_doping_level='+str(x)+', steps_to_burn'+str(steps_to_burn))

if 1: #should each 3d map of the spins be drawn?
	moment_visualization_scale_factor = 0.5
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	#plot solution
	for i in range(0,edge_length):
		for j in range(0,edge_length):
			for k in range(0,edge_length):
				if (i+j+k)%2 == 0:
					plot_color = 'black'
				else:
					plot_color = 'red'
				ax.scatter(i, j, k, color = plot_color, marker='o')
				ax.plot([i,i+s_x[i,j,k]*moment_visualization_scale_factor], [j,j+s_y[i,j,k]*moment_visualization_scale_factor], [k,k+s_z[i,j,k]*moment_visualization_scale_factor], color = plot_color)


plt.show()
