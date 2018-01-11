from __future__ import print_function
from idem_hberg_mc import *

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
		return [La_paper.JFeFeb, La_paper.JFeFeac, La_paper.JMnMnb, La_paper.JMnMnac, La_paper.JMnFeb, La_paper.JMnFeac]
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
Nd_paper.JMnMnac = -1.31 * 11.605 #converting meV to Kelvins
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


temperature_max=200
temperature_min=1.0
temperature_steps=51
equilibration_steps=200
number_of_angle_states=100
magnetic_field=np.array([0.0,0.0,0.0]))
edge_length = 6

x_values_to_run = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
max_temperatures_to_run = [
min_temperatures_to_run

exit()

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

	
my_lattice = SpinLattice(\
iron_doping_level=0.0, edge_length = 6, s_max_0 = 2.5, s_max_1 = 2.0, \
single_ion_anisotropy_len_0 = D_0, single_ion_anisotropy_len_1 = D_1, \
single_ion_anisotropy_hat_0 = D_0_hat, single_ion_anisotropy_hat_1 = D_0_hat, \
superexchange = superexchange_list, \
magnetic_field = np.array([0,0,0]), file_prefix = "NMFO_D_0_DubPerov_", anisotropy_symmetry = "Pnma")
if 1:
	#this is the solid solution
	my_lattice.init_arrays()
if 0:
	#this is the double perovskite
	my_lattice.init_arrays_double_perovskite()
my_lattice.make_op_masks()
my_lattice.bond_list_calc()


my_lattice.random_ijk_list_generator()

my_lattice.temperature_sweep(temperature_max=200, temperature_min=1.0, temperature_steps=51, \
equilibration_steps=150, number_of_angle_states=100, magnetic_field=np.array([0.0,0.0,0.0]))

