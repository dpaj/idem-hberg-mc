from __future__ import print_function
from idem_hberg_mc import *
import numpy as np
import json
import re
import subprocess
import time


def write_then_run(run_parameters_file):
	with open(run_parameters_file) as json_file:  
		run_parameters = json.load(json_file)	
	restart_file_name_to_write = (str("multirestart_"+my_keyword+"_x="+str(run_parameters["iron_doping_level"])\
	+"_L="+str(run_parameters["edge_length"])+".py"))
	
	print(restart_file_name_to_write)
	f = open(restart_file_name_to_write, 'w')
	
	f.write('from __future__ import print_function\n')
	f.write('from idem_hberg_mc import *\n')
	f.write('import numpy as np\n')
	f.write('import json\n')
	f.write('import re\n')
	f.write('run_parameters_file ="'+run_parameters_file+'" \n')
	f.write('\n')
	f.write('file_with_xL = run_parameters_file.split("_run_parameters")[0]\n')
	f.write('print(file_with_xL)\n')
	f.write('\n')
	f.write('with open(run_parameters_file) as json_file:  \n')
	f.write('	run_parameters = json.load(json_file)\n')
	f.write('\n')
	f.write('print(run_parameters)\n')
	f.write('\n')
	f.write('#parameters for the SpinLattice object creation\n')
	f.write('D_0_len = run_parameters["D_0_len"]\n')
	f.write('D_1_len = run_parameters["D_1_len"]\n')
	f.write('D_0_hat = run_parameters["D_0_hat"]\n')
	f.write('D_1_hat = run_parameters["D_1_hat"]\n')
	f.write('iron_doping_level = run_parameters["iron_doping_level"]\n')
	f.write('edge_length = run_parameters["edge_length"]\n')
	f.write('s_max_0 = run_parameters["s_max_0"]\n')
	f.write('s_max_1 = run_parameters["s_max_1"]\n')
	f.write('superexchange = run_parameters["superexchange"]\n')
	f.write('magnetic_field = run_parameters["magnetic_field"]\n')
	f.write('file_prefix = run_parameters["file_prefix"]\n')
	f.write('anisotropy_symmetry = run_parameters["anisotropy_symmetry"]\n')
	f.write('\n')
	f.write('#tempeature sweep parameters\n')
	f.write('double_perovskite = run_parameters["double_perovskite"]\n')
	f.write('temperature_max = run_parameters["temperature_max"]\n')
	f.write('temperature_min = run_parameters["temperature_min"]\n')
	f.write('temperature_steps = run_parameters["temperature_steps"]\n')
	f.write('equilibration_steps = run_parameters["equilibration_steps"]\n')
	f.write('number_of_angle_states = run_parameters["number_of_angle_states"]\n')
	f.write('start_time = run_parameters["start_time"]\n')
	f.write('\n')
	f.write('atom_type = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_atom_type_array.npy"))\n')
	f.write('\n')
	f.write('s_x = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_s_x.npy"))\n')
	f.write('s_y = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_s_y.npy"))\n')
	f.write('s_z = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_s_z.npy"))\n')
	f.write('\n')
	f.write('E_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_E_temperature_array.npy"))\n')
	f.write('A_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_A_temperature_array.npy"))\n')
	f.write('B_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_B_temperature_array.npy"))\n')
	f.write('C_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_C_temperature_array.npy"))\n')
	f.write('G_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_G_temperature_array.npy"))\n')
	f.write('nn_pair_corr_abs_abc_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_nn_pair_corr_abs_abc_temperature_array.npy"))\n')
	f.write('nn_pair_corr_ac_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_nn_pair_corr_ac_temperature_array.npy"))\n')
	f.write('nn_pair_corr_b_temperature_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_nn_pair_corr_b_temperature_array.npy"))\n')
	f.write('temperature_sweep_array = np.load(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_temperature_sweep_array.npy"))\n')
	f.write('\n')
	f.write('\n')
	f.write('# with open(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_run_status.txt")) as json_file:  \n')
	f.write('	# run_parameters = json.load(json_file)\n')
	f.write('\n')
	f.write('\n')
	f.write('run_status_file_data = np.genfromtxt(str(file_prefix+str(int(start_time))+"_x="+str(iron_doping_level)+"_L="+str(edge_length) + "_run_status.txt"), skip_header = 2, delimiter=",")\n')
	f.write('\n')
	f.write('print(run_status_file_data)\n')
	f.write('\n')
	f.write('temperature_last_completed = run_status_file_data[-1][0]\n')
	f.write('\n')
	f.write('my_lattice = SpinLattice(\\\n')
	f.write('iron_doping_level=iron_doping_level, edge_length = edge_length, s_max_0 = s_max_0, s_max_1 = s_max_1, \\\n')
	f.write('single_ion_anisotropy_len_0 = D_0_len, single_ion_anisotropy_len_1 = D_1_len, \\\n')
	f.write('single_ion_anisotropy_hat_0 = D_0_hat, single_ion_anisotropy_hat_1 = D_0_hat, \\\n')
	f.write('superexchange = superexchange, \\\n')
	f.write('magnetic_field = magnetic_field, file_prefix = file_prefix, anisotropy_symmetry = anisotropy_symmetry)')
	f.write('\n')
	f.write('\n')
	f.write('my_lattice.init_arrays_restart(atom_type = atom_type, s_x = s_x, s_y = s_y, s_z = s_z)\n')
	f.write('\n')
	f.write('my_lattice.make_op_masks()\n')
	f.write('my_lattice.bond_list_calc()\n')
	f.write('\n')
	f.write('my_lattice.random_ijk_list_generator()\n')
	f.write('\n')
	f.write('my_lattice.temperature_sweep_restart(temperature_max=temperature_max, temperature_min=temperature_min, temperature_steps=temperature_steps, \\\n')
	f.write('equilibration_steps=equilibration_steps, number_of_angle_states=number_of_angle_states, magnetic_field=magnetic_field, restart_time = start_time, \\\n')
	f.write('E_temperature_array = E_temperature_array, A_temperature_array = A_temperature_array, B_temperature_array = B_temperature_array, C_temperature_array = C_temperature_array, G_temperature_array = G_temperature_array, \\\n')
	f.write('nn_pair_corr_abs_abc_temperature_array = nn_pair_corr_abs_abc_temperature_array, nn_pair_corr_ac_temperature_array = nn_pair_corr_ac_temperature_array, nn_pair_corr_b_temperature_array = nn_pair_corr_b_temperature_array, \\\n')
	f.write('temperature_sweep_array = temperature_sweep_array, temperature_last_completed = temperature_last_completed)\n')
	subprocess.call(r"nohup python "+restart_file_name_to_write+" & disown", shell=True, stdout=subprocess.PIPE)



my_keyword = "LMFO_MnFe=G"

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
	time.sleep(1)
