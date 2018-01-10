from __future__ import print_function
from idem_hberg_mc import *

print(time()-start_time)
#type 0 = Fe
#type 1 = Mn
#order of superexchange list
# JFeFeb, JFeFeac, JMnMnb, JMnMnac, JMnFeb, JMnFeac
#to use the hardcoded DFT values for La, just put -1
#to use the hardcoded DFT values for Nd, just put -2
#to use the experimental values for La and old DFT for MnFe, just put -3
#to use the hardcoded extrapolated values for La, just put -4
#to use the hardcoded extrapolated values for Nd, just put -5
#REMEMBER THE SIGN CONVENTION FOR J, here the J>0 is antiferro J< is ferro

D_0 = 0.02 #Fe eventually should be in Kelvins for the code
D_1 = -1.85673417 #Mn eventually should be in Kelvins for the code

D_0_hat = np.array([1.882,1.2248,0.332])
D_1_hat = np.array([1.882,1.2248,0.332])

D_0_hat = D_0_hat / np.sqrt(np.dot(D_0_hat, D_0_hat))
D_1_hat = D_1_hat / np.sqrt(np.dot(D_1_hat, D_1_hat))
	
my_lattice = SpinLattice(\
iron_doping_level=0.0, edge_length = 6, s_max_0 = 2.5, s_max_1 = 2.0, \
single_ion_anisotropy_len_0 = D_0, single_ion_anisotropy_len_1 = D_1, \
single_ion_anisotropy_hat_0 = D_0_hat, single_ion_anisotropy_hat_1 = D_0_hat, \
superexchange = -4, \
magnetic_field = np.array([0,0,0]), file_prefix = "NMFO_D_0_DubPerov_", anisotropy_symmetry = "Pnma")
if 1:
	#this is the solid solution
	my_lattice.init_arrays()
if 0:
	#this is the double perovskite
	my_lattice.init_arrays_double_perovskite()
my_lattice.make_op_masks()
my_lattice.bond_list_calc()


print(time()-start_time)
my_lattice.random_ijk_list_generator()

my_lattice.temperature_sweep(temperature_max=200, temperature_min=1.0, temperature_steps=51, \
equilibration_steps=150, number_of_angle_states=100, magnetic_field=np.array([0.0,0.0,0.0]))

print('\ntime=', time()-start_time)
