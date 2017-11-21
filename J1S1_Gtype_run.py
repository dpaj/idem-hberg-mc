from __future__ import print_function
from idem_hberg_mc import *

print(time()-start_time)
#type 0 = Fe
#type 1 = Mn
	
my_lattice = SpinLattice(\
iron_doping_level=0.0, edge_length = 8, s_max_0 = 1.0, s_max_1 = 1.0, \
single_ion_anisotropy_0 = np.array([0,0,0]), single_ion_anisotropy_1 = np.array([0,-0.5,0.0]), superexchange = [1,1,1,1,1,1], \
magnetic_field = np.array([0,0,0]), file_prefix = "J1S1Dm0p5y_")
my_lattice.init_arrays()
my_lattice.make_op_masks()
my_lattice.bond_list_calc()


print(time()-start_time)
my_lattice.random_ijk_list_generator()

my_lattice.temperature_sweep(temperature_max=2.501, temperature_min=1e-3, temperature_steps=26, \
equilibration_steps=50, number_of_angle_states=100, magnetic_field=np.array([0.0,0.0,0.0]))

print('\ntime=', time()-start_time)