from __future__ import print_function
from idem_hberg_mc import *

print(time()-start_time)
#type 0 = Fe
#type 1 = Mn
#order of superexchange list
# JFeFeb, JFeFeac, JMnMnb, JMnMnac, JMnFeb, JMnFeac
#to use the hardcoded values for La, just put -1
#to use the hardcoded values for Nd, just put -2
#to use the experimental values for La and old DFT for MnFe, just put -3
	
my_lattice = SpinLattice(\
iron_doping_level=0.5, edge_length = 22, s_max_0 = 2.5, s_max_1 = 2.0, \
single_ion_anisotropy_0 = np.array([0,0,-0.0]), single_ion_anisotropy_1 = np.array([-0.0,0,0.0]), superexchange = -2, \
magnetic_field = np.array([0,0,0]), file_prefix = "NMFO_D_0_DubPerov_")
if 0:
	#this is the solid solution
	my_lattice.init_arrays()
if 1:
	#this is the double perovskite
	my_lattice.init_arrays_double_perovskite()
my_lattice.make_op_masks()
my_lattice.bond_list_calc()


print(time()-start_time)
my_lattice.random_ijk_list_generator()

my_lattice.temperature_sweep(temperature_max=801.0, temperature_min=1.0, temperature_steps=81, \
equilibration_steps=200, number_of_angle_states=100, magnetic_field=np.array([0.0,0.0,0.0]))

print('\ntime=', time()-start_time)
