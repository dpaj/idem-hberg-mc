from __future__ import print_function
from idem_hberg_mc import *
import os
import subprocess
import numpy as np
import time

"""			
JFeFeb = self.superexchange[0]
JFeFeac = self.superexchange[1]
JMnMnb = self.superexchange[2]
JMnMnac = self.superexchange[3]
JMnFeb = self.superexchange[4]
JMnFeac = self.superexchange[5]
"""
class ParameterList(object):
	JFeFeb = -99999999999999.0
	JFeFeac = -99999999999999.0
	JMnMnb = -99999999999999.0
	JMnMnac = -99999999999999.0
	JMnFeb = -99999999999999.0
	JMnFeac = -99999999999999.0

	# The class "constructor" - It's actually an initializer 
	def __init__(self, JFeFeb=None, JFeFeac=None, JMnMnb=None, JMnMnac=None, JMnFeb=None, JMnFeac=None, DMn_len=None, DMn_hat=None, DFe_len=None, DFe_hat=None):
		self.JFeFeb = JFeFeb
		self.JFeFeac = JFeFeac
		self.JMnMnb = JMnMnb
		self.JMnMnac = JMnMnac
		self.JMnFeb = JMnFeb
		self.JMnFeac = JMnFeac
		self.DMn_len = DMn_len
		self.DMn_hat = DMn_hat
		self.DFe_len = DFe_len
		self.DFe_hat = DFe_hat
	def return_superexchange_list(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, self.JMnFeb, self.JMnFeac]
	def return_superexchange_list_MnFeA(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, self.JMnFeb, -self.JMnFeac]
	def return_superexchange_list_MnFeC(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, -self.JMnFeb, self.JMnFeac]
	def return_superexchange_list_MnFeB(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, -self.JMnFeb, -self.JMnFeac]
	def return_superexchange_list_MnFeG(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, self.JMnFeb, self.JMnFeac]
	def return_superexchange_list_MnFe0(self):
		return [self.JFeFeb, self.JFeFeac, self.JMnMnb, self.JMnMnac, 0.0,0.0]
	def normalize_hat(self):
		DFe_hat = self.DFe_hat
		DMn_hat = self.DMn_hat

		DFe_hat = DFe_hat / np.sqrt(np.dot(DFe_hat, DFe_hat))
		DMn_hat = DMn_hat / np.sqrt(np.dot(DMn_hat, DMn_hat))

		self.DFe_hat = DFe_hat
		self.DMn_hat = DMn_hat		
		
		
#JFeFeb, JFeFeac, JMnMnb, JMnMnac, JMnFeb, JMnFeac
def make_parameter_list():
    superexchange = ParameterList()
    return superexchange
	

def write_and_run_file(iron_doping_level, edge_length, \
single_ion_anisotropy_len_0, single_ion_anisotropy_len_1, single_ion_anisotropy_hat_0, single_ion_anisotropy_hat_1, \
superexchange_list, \
file_prefix, \
anisotropy_symmetry, \
temperature_max, temperature_min, temperature_steps, equilibration_steps):
	iron_doping_level = str(iron_doping_level)
	edge_length = str(edge_length)
	single_ion_anisotropy_len_0 = str(single_ion_anisotropy_len_0)
	single_ion_anisotropy_len_1 = str(single_ion_anisotropy_len_1)
	single_ion_anisotropy_hat_0 = np.array2string(single_ion_anisotropy_hat_0, separator=', ')
	single_ion_anisotropy_hat_1 = np.array2string(single_ion_anisotropy_hat_1, separator=', ')
	superexchange_list = str(superexchange_list)
	file_prefix_no_quotes = file_prefix
	file_prefix = '"'+str(file_prefix)+'"'
	anisotropy_symmetry = '"'+str(anisotropy_symmetry)+'"'
	temperature_max = str(temperature_max)
	temperature_min = str(temperature_min)
	temperature_steps = str(temperature_steps)
	equilibration_steps = str(equilibration_steps)
	if os.name == 'nt':
		the_file_name_used = "multirun_"+file_prefix_no_quotes+"x="+iron_doping_level+"_L="+edge_length+".py"
		print(the_file_name_used)
		f = open("dummy.py", 'w')
	elif os.name == 'posix':
		the_file_name_used = "multirun_"+file_prefix_no_quotes+"x="+iron_doping_level+"_L="+edge_length+".py"
		f = open(str(the_file_name_used), 'w')
	
	f.write('from idem_hberg_mc import *')
	f.write('\n')
	f.write('#type 0 = Fe\n')
	f.write('#type 1 = Mn\n')
	f.write('#order of superexchange list\n')
	f.write('# JFeFeb, JFeFeac, JMnMnb, JMnMnac, JMnFeb, JMnFeac\n')
	f.write('#to use the hardcoded DFT values for La, just put -1\n')
	f.write('#to use the hardcoded DFT values for Nd, just put -2\n')
	f.write('#to use the experimental values for La and old DFT for MnFe, just put -3\n')
	f.write('#to use the hardcoded extrapolated values for La, just put -4\n')
	f.write('#to use the hardcoded extrapolated values for Nd, just put -5\n')
	f.write('#REMEMBER THE SIGN CONVENTION FOR J, here the J>0 is antiferro J< is ferro\n')
	f.write('\n')
	f.write('\n')
	f.write('my_lattice = SpinLattice(\\\n')
	f.write('iron_doping_level='+iron_doping_level+', edge_length = '+edge_length+', s_max_0 = 2.5, s_max_1 = 2.0, \\\n')
	f.write('single_ion_anisotropy_len_0 = '+single_ion_anisotropy_len_0+', single_ion_anisotropy_len_1 = '+single_ion_anisotropy_len_1+', \\\n')
	f.write('single_ion_anisotropy_hat_0 = '+single_ion_anisotropy_hat_0+', single_ion_anisotropy_hat_1 = '+single_ion_anisotropy_hat_1+', \\\n')
	f.write('superexchange = '+superexchange_list+', \\\n')
	f.write('magnetic_field = np.array([0,0,0]), file_prefix = '+file_prefix+', anisotropy_symmetry = '+anisotropy_symmetry+')\n')
	f.write('if 1:\n')
	f.write('	#this is the solid solution\n')
	f.write('	my_lattice.init_arrays()\n')
	f.write('if 0:\n')
	f.write('	#this is the double perovskite\n')
	f.write('	my_lattice.init_arrays_double_perovskite()\n')
	f.write('my_lattice.make_op_masks()\n')
	f.write('my_lattice.bond_list_calc()\n')
	f.write('\n')
	f.write('\n')
	f.write('my_lattice.random_ijk_list_generator()\n')
	f.write('\n')
	f.write('my_lattice.temperature_sweep(temperature_max='+temperature_max+', temperature_min='+temperature_min+', temperature_steps='+temperature_steps+', \\\n')
	f.write('equilibration_steps='+equilibration_steps+', number_of_angle_states=100, magnetic_field=np.array([0.0,0.0,0.0]))\n')
	f.close()


#La values from my paper
La_paper = make_parameter_list()
La_paper.JFeFeb = 6.34 * 11.605 #converting meV to Kelvins
La_paper.JFeFeac = 7.05 * 11.605 #converting meV to Kelvins
La_paper.JMnMnb = 1.55 * 11.605 #converting meV to Kelvins
La_paper.JMnMnac = -2.12 * 11.605 #converting meV to Kelvins
La_paper.JMnFeb = 0.66 * 11.605 #converting meV to Kelvins
La_paper.JMnFeac = 4.35 * 11.605 #converting meV to Kelvins
La_paper.DMn_len = -0.160* 11.605
La_paper.DMn_hat = np.array([1.882,1.2248,0.332])
La_paper.DFe_len = 0.002* 11.605
La_paper.DFe_hat = np.array([1.882,1.2248,0.332])
La_paper.normalize_hat()

#Nd values from my paper
Nd_paper = make_parameter_list()
Nd_paper.JFeFeb = 5.93 * 11.605 #converting meV to Kelvins
Nd_paper.JFeFeac = 6.57 * 11.605 #converting meV to Kelvins
Nd_paper.JMnMnb = 1.55 * 11.605 #converting meV to Kelvins
Nd_paper.JMnMnac = -0.95 * 11.605 #converting meV to Kelvins
Nd_paper.JMnFeb = 0.87 * 11.605 #converting meV to Kelvins
Nd_paper.JMnFeac = 4.16 * 11.605 #converting meV to Kelvins
Nd_paper.DMn_len = -0.158* 11.605
Nd_paper.DMn_hat = np.array([1.882,1.2248,0.332])
Nd_paper.DFe_len = 0.004* 11.605
Nd_paper.DFe_hat = np.array([1.882,1.2248,0.332])
Nd_paper.normalize_hat()
print(Nd_paper.DMn_len)

superexchange_list = La_paper.return_superexchange_list()
print(superexchange_list)

single_ion_anisotropy_len_0, single_ion_anisotropy_len_1, single_ion_anisotropy_hat_0, single_ion_anisotropy_hat_1 = La_paper.DFe_len, La_paper.DMn_len, La_paper.DFe_hat, La_paper.DMn_hat

temperature_steps=101
equilibration_steps=200
number_of_angle_states=100
magnetic_field=np.array([0.0,0.0,0.0])
edge_length = 22

anisotropy_symmetry = "Pnma"

TN_LMO = 138.0
TN_NMO = 82.0
TN_LFO = 738.0
TN_NFO = 689.0

x_values_to_run = np.linspace(0,1,11)
max_temperatures_to_run = np.linspace(TN_LMO+50, TN_LFO+100,11)
min_temperatures_to_run = np.ones(11)

print(x_values_to_run, max_temperatures_to_run, min_temperatures_to_run)
wait_between_file_runs = 0#seconds
file_prefix_list = ["LMFO_MnFe=0_"]
for sx_fn_idx, superexchange_function in enumerate([La_paper.return_superexchange_list_MnFe0]):
	superexchange_list = superexchange_function()
	file_prefix = file_prefix_list[sx_fn_idx]
	for idx, iron_doping_level in enumerate(x_values_to_run):
		write_and_run_file(iron_doping_level, edge_length, \
		single_ion_anisotropy_len_0, single_ion_anisotropy_len_1, single_ion_anisotropy_hat_0, single_ion_anisotropy_hat_1, \
		superexchange_list, \
		file_prefix, \
		anisotropy_symmetry, \
		max_temperatures_to_run[idx], min_temperatures_to_run[idx], temperature_steps, equilibration_steps)
		
		time.sleep(wait_between_file_runs)


