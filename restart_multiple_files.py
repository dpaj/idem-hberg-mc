from __future__ import print_function
from idem_hberg_mc import *
import numpy as np
import json
import re


def write_then_run(run_parameters_file):
	with open(run_parameters_file) as json_file:  
		run_parameters = json.load(json_file)	
	restart_file_name_to_write = (str("multirestart_"+my_keyword+"_x="+str(run_parameters["iron_doping_level"])\
	+"_L="+str(run_parameters["edge_length"])+".py"))
	
	
	f = open(restart_file_name_to_write, 'w')
	
	f.write('from __future__ import print_function')
	f.write('from idem_hberg_mc import *')
	f.write('import numpy as np')
	f.write('import json')
	f.write('import re')
	f.write('run_parameters_file ='+run_parameters_file)
	exit()
	file_with_xL = run_parameters_file.split('_run_parameters')[0]
	print(file_with_xL)

	with open(run_parameters_file) as json_file:  
		run_parameters = json.load(json_file)

	print(run_parameters)

	#parameters for the SpinLattice object creation
	D_0_len = run_parameters["D_0_len"]
	D_1_len = run_parameters["D_1_len"]
	D_0_hat = run_parameters["D_0_hat"]
	D_1_hat = run_parameters["D_1_hat"]
	iron_doping_level = run_parameters["iron_doping_level"]
	edge_length = run_parameters["edge_length"]
	s_max_0 = run_parameters["s_max_0"]
	s_max_1 = run_parameters["s_max_1"]
	superexchange = run_parameters["superexchange"]
	magnetic_field = run_parameters["magnetic_field"]
	file_prefix = run_parameters["file_prefix"]
	anisotropy_symmetry = run_parameters["anisotropy_symmetry"]

	#tempeature sweep parameters
	double_perovskite = run_parameters["double_perovskite"]
	temperature_max = run_parameters["temperature_max"]
	temperature_min = run_parameters["temperature_min"]
	temperature_steps = run_parameters["temperature_steps"]
	equilibration_steps = run_parameters["equilibration_steps"]
	number_of_angle_states = run_parameters["number_of_angle_states"]
	start_time = run_parameters["start_time"]

	atom_type = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_atom_type_array.npy"))

	s_x = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_s_x.npy"))
	s_y = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_s_y.npy"))
	s_z = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_s_z.npy"))

	E_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_E_temperature_array.npy"))
	A_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_A_temperature_array.npy"))
	B_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_B_temperature_array.npy"))
	C_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_C_temperature_array.npy"))
	G_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_G_temperature_array.npy"))
	nn_pair_corr_abs_abc_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_nn_pair_corr_abs_abc_temperature_array.npy"))
	nn_pair_corr_ac_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_nn_pair_corr_ac_temperature_array.npy"))
	nn_pair_corr_b_temperature_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_nn_pair_corr_b_temperature_array.npy"))
	temperature_sweep_array = np.load(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_temperature_sweep_array.npy"))


	# with open(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_run_status.txt")) as json_file:  
		# run_parameters = json.load(json_file)


	run_status_file_data = np.genfromtxt(str(file_prefix+str(int(start_time))+'_x='+str(iron_doping_level)+'_L='+str(edge_length) + "_run_status.txt"), skip_header = 2, delimiter=',')

	print(run_status_file_data)

	temperature_last_completed = run_status_file_data[-1][0]

	my_lattice = SpinLattice(\
	iron_doping_level=iron_doping_level, edge_length = edge_length, s_max_0 = s_max_0, s_max_1 = s_max_1, \
	single_ion_anisotropy_len_0 = D_0_len, single_ion_anisotropy_len_1 = D_1_len, \
	single_ion_anisotropy_hat_0 = D_0_hat, single_ion_anisotropy_hat_1 = D_0_hat, \
	superexchange = superexchange, \
	magnetic_field = magnetic_field, file_prefix = file_prefix, anisotropy_symmetry = anisotropy_symmetry)

	my_lattice.init_arrays_restart(atom_type = atom_type, s_x = s_x, s_y = s_y, s_z = s_z)

	my_lattice.make_op_masks()
	my_lattice.bond_list_calc()

	my_lattice.random_ijk_list_generator()

	my_lattice.temperature_sweep_restart(temperature_max=temperature_max, temperature_min=temperature_min, temperature_steps=temperature_steps, \
	equilibration_steps=equilibration_steps, number_of_angle_states=number_of_angle_states, magnetic_field=magnetic_field, restart_time = start_time, \
	E_temperature_array = E_temperature_array, A_temperature_array = A_temperature_array, B_temperature_array = B_temperature_array, C_temperature_array = C_temperature_array, G_temperature_array = G_temperature_array, \
	nn_pair_corr_abs_abc_temperature_array = nn_pair_corr_abs_abc_temperature_array, nn_pair_corr_ac_temperature_array = nn_pair_corr_ac_temperature_array, nn_pair_corr_b_temperature_array = nn_pair_corr_b_temperature_array, \
	temperature_sweep_array = temperature_sweep_array, temperature_last_completed = temperature_last_completed)




my_keyword = "NMFO_MnFe=0"

files = os.listdir('.')
json_files = []
for i in files:
	if (re.search("json", i)):
		json_files.append(i)

		
json_files.sort()

files_to_run = []
for i in json_files:
	if (re.search(my_keyword, i)):
		files_to_run.append(i)

for i in files_to_run:
	print(i)
	write_then_run(i)
