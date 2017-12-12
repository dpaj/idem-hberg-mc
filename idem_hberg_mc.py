from __future__ import print_function
from time import time
#for timing of simulation
start_time = time()

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as spop
from scipy.integrate import quad
from scipy.interpolate import interp1d
import random
import os


hbar = 6.626070040e-34 #SI units

#arbitrary number visualize spins
moment_visualization_scale_factor = 0.1


class SpinLattice(object):
	""" this class is defined to house the lattice parameters and variables
	
	s_x, s_y, s_z = these are edge_length**3 arrays of the cartesian components of the spins
	phi = this is an edge_length**3 array of the azimuthal angles, each of which has a range of [0,2*pi]
	theta = this is an edge_length**3 array of the elevation angle, each of which has a range of [0,pi]
	energy = this is an edge_length**3 array of the site energies
	
	iron_doping_level = varies from 0 to 1, 0 is no iron, is set for a given SpinLattice object
	edge_length = the simulation is the edge of a cube with periodic boundary conditions, and this is just the number of sites per side
	single_ion_anisotropy = this is the 6-d array, where the first three dimensions correspond to indexation of sites within the simulated lattice, and the final three dimensions house the direction and strength of the single ion anisotropy for a given site
	
	single_ion_anisotropy_0 = the is the 3-d array that holds the strength and direction of the iron single ion anisotropy
	single_ion_anisotropy_1 = the 3-d array that holds the strength and direction of the manganese single ion anisotropy
	
	s_max_0 = the double that holds the maximum spin value of the iron spin
	s_max_1 = the double that holds the maximum spin value of the manganese spin
	
	g_type_mask = the edge_length**3 array that is masked to be g-type for all sites
	a_type_mask = the edge_length**3 array that is masked to be a-type for all sites
	mn_g_type_mask = the edge_length**3 array that is masked to be g-type for ONLY the mn sites
	mn_a_type_mask = the edge_length**3 array that is masked to be a-type for ONLY the mn sites
	fe_g_type_mask = the edge_length**3 array that is masked to be g-type for ONLY the fe sites
	fe_a_type_mask = the edge_length**3 array that is masked to be a-type for ONLY the fe sites	
	
	superexchange_array = the 9-d array that houses the superexchange interactions for the solid solution, such that the first 3-d are indexation of the sites within the simulated lattice, and the final six dimensions are the nearest neighbor interactions
	
	total_energy = ???
	
	random_ijk_list = when updating, the order is random but the same every update cycle, and this list stores a series of indices


	"""
	def __init__(self, edge_length=None, iron_doping_level=None, s_max_0=None, s_max_1=None, single_ion_anisotropy_0 = None, single_ion_anisotropy_1 = None, single_ion_anisotropy=None, superexchange=None, magnetic_field=None, s_x=None, s_y=None, s_z=None, phi=None, theta=None, energy=None, total_energy=None, random_ijk_array=None,temporary_nn_pair_corr=None, atom_type=None, file_prefix=""):
		self.iron_doping_level = iron_doping_level
		self.edge_length = edge_length
		self.single_ion_anisotropy = np.zeros((edge_length,edge_length,edge_length,3))
		self.single_ion_anisotropy_0 = single_ion_anisotropy_0
		self.single_ion_anisotropy_1 = single_ion_anisotropy_1
		self.s_max_0 = s_max_0
		self.s_max_1 = s_max_1
		self.g_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.a_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.mn_g_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.mn_a_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.fe_g_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.fe_a_type_mask = np.zeros((edge_length,edge_length,edge_length))
		self.superexchange = superexchange
		self.superexchange_array = np.zeros((edge_length,edge_length,edge_length,6))
		self.magnetic_field = magnetic_field
		self.s_max = np.zeros((edge_length,edge_length,edge_length))
		self.s_x = np.zeros((edge_length,edge_length,edge_length))
		self.s_y = np.zeros((edge_length,edge_length,edge_length))
		self.s_z = np.zeros((edge_length,edge_length,edge_length))
		self.phi = np.zeros((edge_length,edge_length,edge_length))
		self.theta = np.zeros((edge_length,edge_length,edge_length))
		self.energy = np.zeros((edge_length,edge_length,edge_length))
		self.total_energy = 0
		self.random_ijk_list = []
		self.temporary_nn_pair_corr = 0
		self.atom_type = np.zeros((edge_length,edge_length,edge_length), dtype = np.int8)
		self.file_prefix = file_prefix
		
	def __str__(self):
		return "SpinLattice"
	def get_edge_length(self):
		return self.edge_length
	def get_s_max(self):
		return self.s_max
	def get_single_ion_anisotropy(self):
		return self.single_ion_anisotropy
	def get_superexchange(self):
		return self.superexchange
	def get_magnetic_field(self):
		return self.magnetic_field
	def get_s_x(self):
		return self.s_x
	def get_s_y(self):
		return self.s_y
	def get_s_z(self):
		return self.s_z
	def get_phi(self):
		return self.phi
	def get_theta(self):
		return self.theta
	def get_energy(self):
		return self.energy


	def energy_calc(self, x, super_exchange_field, single_ion_anisotropy_ijk, s_max_ijk):
		
		#s_max = self.s_max
		#print(s_max)
		theta, phi = x[0], x[1]
		s_x_ijk = s_max_ijk*np.sin(theta)*np.cos(phi)
		s_y_ijk = s_max_ijk*np.sin(theta)*np.sin(phi)
		s_z_ijk = s_max_ijk*np.cos(theta)
		s_vec_ijk = np.array([s_x_ijk, s_y_ijk, s_z_ijk])
		
		energy_ijk = np.dot(super_exchange_field, s_vec_ijk) + s_x_ijk**2*single_ion_anisotropy_ijk[0] + s_y_ijk**2*single_ion_anisotropy_ijk[1] + s_z_ijk**2*single_ion_anisotropy_ijk[2]
		
		return energy_ijk
	def energy_calc_simple(self, x, i, j, k):
		s_max = self.s_max
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		theta, phi = x[0], x[1]
		s_x_ijk = s_max[i,j,k]*np.sin(theta)*np.cos(phi)
		s_y_ijk = s_max[i,j,k]*np.sin(theta)*np.sin(phi)
		s_z_ijk = s_max[i,j,k]*np.cos(theta)
		s_vec = np.array([s_x_ijk, s_y_ijk, s_z_ijk])
		energy_ijk = 0
		#superexchange = self.superexchange
		superexchange_array = self.superexchange_array
		superexchange = superexchange_array[i,j,k]
		single_ion_anisotropy = self.single_ion_anisotropy

		energy_ijk += np.dot(single_ion_anisotropy[i,j,k], s_vec)

		if i < edge_length-1:
			energy_ijk += superexchange[0]*(s_x_ijk*s_x[i+1,j,k] + s_y_ijk*s_y[i+1,j,k] + s_z_ijk*s_z[i+1,j,k])
		else:
			energy_ijk += superexchange[0]*(s_x_ijk*s_x[0,j,k] + s_y_ijk*s_y[0,j,k] + s_z_ijk*s_z[0,j,k])
		if i > 0:
			energy_ijk += superexchange[1]*(s_x_ijk*s_x[i-1,j,k] + s_y_ijk*s_y[i-1,j,k] + s_z_ijk*s_z[i-1,j,k])
		else:
			energy_ijk += superexchange[1]*(s_x_ijk*s_x[edge_length-1,j,k] + s_y_ijk*s_y[edge_length-1,j,k] + s_z_ijk*s_z[edge_length-1,j,k])
			
		if j < edge_length-1:
			energy_ijk += superexchange[2]*(s_x_ijk*s_x[i,j+1,k] + s_y_ijk*s_y[i,j+1,k] + s_z_ijk*s_z[i,j+1,k])
		else:
			energy_ijk += superexchange[2]*(s_x_ijk*s_x[i,0,k] + s_y_ijk*s_y[i,0,k] + s_z_ijk*s_z[i,0,k])
		if j > 0:
			energy_ijk += superexchange[3]*(s_x_ijk*s_x[i,j-1,k] + s_y_ijk*s_y[i,j-1,k] + s_z_ijk*s_z[i,j-1,k])
		else:
			energy_ijk += superexchange[3]*(s_x_ijk*s_x[i,edge_length-1,k] + s_y_ijk*s_y[i,edge_length-1,k] + s_z_ijk*s_z[i,edge_length-1,k])
			
		if k < edge_length-1:
			energy_ijk += superexchange[4]*(s_x_ijk*s_x[i,j,k+1] + s_y_ijk*s_y[i,j,k+1] + s_z_ijk*s_z[i,j,k+1])
		else:
			energy_ijk += superexchange[4]*(s_x_ijk*s_x[i,j,0] + s_y_ijk*s_y[i,j,0] + s_z_ijk*s_z[i,j,0])
		if k > 0:
			energy_ijk += superexchange[5]*(s_x_ijk*s_x[i,j,k-1] + s_y_ijk*s_y[i,j,k-1] + s_z_ijk*s_z[i,j,k-1])
		else:
			energy_ijk += superexchange[5]*(s_x_ijk*s_x[i,j,edge_length-1] + s_y_ijk*s_y[i,j,edge_length-1] + s_z_ijk*s_z[i,j,edge_length-1])
		return energy_ijk
	def superexchange_array_calc(self,i,j,k):
		atom_type = self.atom_type
		#print(atom_type)
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		superexchange_array = self.superexchange_array
		super_exchange_ijk = superexchange_array[i,j,k] #np.array([0,0,0,0,0,0])
		atom_type_ijk = atom_type[i,j,k]
		
		if self.superexchange == -3: #pseudo-experimental values
			JFeFeb = 62.0
			JFeFeac = 62.0
			JMnMnb = 6.7
			JMnMnac = -9.6
			JMnFeb = 17.0
			JMnFeac = 29.0
		if self.superexchange == -2: #Nd DFT values
			JFeFeb = 46.3
			JFeFeac = 51.3
			JMnMnb = 3.13
			JMnMnac = -0.58
			JMnFeb = 7.54
			JMnFeac = 35.742
		if self.superexchange == -1: #La DFT values
			JFeFeb = 47.6
			JFeFeac = 53.0
			JMnMnb = -3.01
			JMnMnac = 3.133
			JMnFeb = 5.686
			JMnFeac = 37.36
		else:
			JFeFeb = self.superexchange[0]
			JFeFeac = self.superexchange[1]
			JMnMnb = self.superexchange[2]
			JMnMnac = self.superexchange[3]
			JMnFeb = self.superexchange[4]
			JMnFeac = self.superexchange[5]

		JFeFex = JFeFeac
		JFeFey = JFeFeb
		JFeFez = JFeFeac

		JMnMnx = JMnMnac
		JMnMny = JMnMnb
		JMnMnz = JMnMnac

		JMnFex = JMnFeac
		JMnFey = JMnFeb
		JMnFez = JMnFeac


		def superexchange_x(type1,type2):
			if type1 == 0 and type2 == 0: #Fe-Fe
				return JFeFex
			elif type1 == 1 and type2 == 1: #Mn-Mn
				return JMnMnx
			else:
				return JMnFex

		def superexchange_y(type1,type2):
			if type1 == 0 and type2 == 0: #Fe-Fe
				return JFeFey
			elif type1 == 1 and type2 == 1: #Mn-Mn
				return JMnMny
			else:
				return JMnFey

		def superexchange_z(type1,type2):
			if type1 == 0 and type2 == 0: #Fe-Fe
				return JFeFez
			elif type1 == 1 and type2 == 1: #Mn-Mn
				return JMnMnz
			else:
				return JMnFez
		
		if i < edge_length-1:
			super_exchange_ijk[0] = superexchange_x(atom_type_ijk, atom_type[i+1,j,k])
		else:
			super_exchange_ijk[0] = superexchange_x(atom_type_ijk, atom_type[0,j,k])
		if i > 0:
			super_exchange_ijk[1] = superexchange_x(atom_type_ijk, atom_type[i-1,j,k])
		else:
			super_exchange_ijk[1] = superexchange_x(atom_type_ijk, atom_type[edge_length-1,j,k])
			
		if j < edge_length-1:
			super_exchange_ijk[2] = superexchange_y(atom_type_ijk, atom_type[i,j+1,k])
		else:
			super_exchange_ijk[2] = superexchange_y(atom_type_ijk, atom_type[i,0,k])
		if j > 0:
			super_exchange_ijk[3] = superexchange_y(atom_type_ijk, atom_type[i,j-1,k])
		else:
			super_exchange_ijk[3] = superexchange_y(atom_type_ijk, atom_type[i,edge_length-1,k])
			
		if k < edge_length-1:
			super_exchange_ijk[4] = superexchange_z(atom_type_ijk, atom_type[i,j,k+1])
		else:
			super_exchange_ijk[4] = superexchange_z(atom_type_ijk, atom_type[i,j,0])
		if k > 0:
			super_exchange_ijk[5] = superexchange_z(atom_type_ijk, atom_type[i,j,k-1])
		else:
			super_exchange_ijk[5] = superexchange_z(atom_type_ijk, atom_type[i,j,edge_length-1])
		return super_exchange_ijk
		
	def init_arrays(self):
		iron_doping_level = self.iron_doping_level
		edge_length, s_x, s_y, s_z, s_max, phi, theta, energy, atom_type = self.edge_length, self.s_x, self.s_y, self.s_z, self.s_max, self.phi, self.theta, self.energy, self.atom_type
		single_ion_anisotropy, single_ion_anisotropy_0, single_ion_anisotropy_1 = self.single_ion_anisotropy, self.single_ion_anisotropy_0, self.single_ion_anisotropy_1
		superexchange_array = self.superexchange_array
		s_max_0, s_max_1 = self.s_max_0, self.s_max_1
		superexchange_array_calc = self.superexchange_array_calc
		single_ion_anisotropy_list = [single_ion_anisotropy_0, single_ion_anisotropy_1]
		s_max_list = [s_max_0, s_max_1]
		#initialize the spin momentum vectors to have a random direction
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					atom_type[i,j,k] = int(np.random.choice([0,1],p=[iron_doping_level, 1.0-iron_doping_level]))
					#print(atom_type[i,j,k])
					#print(single_ion_anisotropy_list[atom_type[i,j,k]])
					single_ion_anisotropy[i,j,k] = single_ion_anisotropy_list[atom_type[i,j,k]]
					s_max[i,j,k] = s_max_list[atom_type[i,j,k]]
					
					
					
					phi[i,j,k] = np.random.rand()*2*np.pi
					theta[i,j,k] = np.arccos(1.0-2.0*np.random.rand())# asdf
					s_x[i,j,k] = s_max[i,j,k]*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
					s_y[i,j,k] = s_max[i,j,k]*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
					s_z[i,j,k] = s_max[i,j,k]*np.cos(theta[i,j,k])
					energy[i,j,k] = self.energy_calc_simple((theta[i,j,k],phi[i,j,k]),i,j,k)
					
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):					
					superexchange_array[i,j,k] = superexchange_array_calc(i,j,k)
		#print('superexchange_array',superexchange_array)
					

		return s_x, s_y, s_z, phi, theta, energy
	def init_arrays_double_perovskite(self):
		iron_doping_level = self.iron_doping_level
		edge_length, s_x, s_y, s_z, s_max, phi, theta, energy, atom_type = self.edge_length, self.s_x, self.s_y, self.s_z, self.s_max, self.phi, self.theta, self.energy, self.atom_type
		single_ion_anisotropy, single_ion_anisotropy_0, single_ion_anisotropy_1 = self.single_ion_anisotropy, self.single_ion_anisotropy_0, self.single_ion_anisotropy_1
		superexchange_array = self.superexchange_array
		s_max_0, s_max_1 = self.s_max_0, self.s_max_1
		superexchange_array_calc = self.superexchange_array_calc
		single_ion_anisotropy_list = [single_ion_anisotropy_0, single_ion_anisotropy_1]
		s_max_list = [s_max_0, s_max_1]
		#initialize the spin momentum vectors to have a random direction
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					if (i+j+k)%2 == 0:
						atom_type[i,j,k] = 0
					else:
						atom_type[i,j,k] = 1
					#print(atom_type[i,j,k])
					#print(single_ion_anisotropy_list[atom_type[i,j,k]])
					single_ion_anisotropy[i,j,k] = single_ion_anisotropy_list[atom_type[i,j,k]]
					s_max[i,j,k] = s_max_list[atom_type[i,j,k]]
					
					
					
					phi[i,j,k] = np.random.rand()*2*np.pi
					theta[i,j,k] = np.arccos(1.0-2.0*np.random.rand())# asdf
					s_x[i,j,k] = s_max[i,j,k]*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
					s_y[i,j,k] = s_max[i,j,k]*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
					s_z[i,j,k] = s_max[i,j,k]*np.cos(theta[i,j,k])
					energy[i,j,k] = self.energy_calc_simple((theta[i,j,k],phi[i,j,k]),i,j,k)
					
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):					
					superexchange_array[i,j,k] = superexchange_array_calc(i,j,k)
		#print('superexchange_array',superexchange_array)
					

		return s_x, s_y, s_z, phi, theta, energy		
	def negative_of_energy_calc(self, x, super_exchange_field):
		return -self.energy_calc(x,super_exchange_field)
	def total_energy_calc(self):
		self.total_energy = np.sum(self.energy)
		return total_energy

	def temperature_sweep(self, temperature_max, temperature_min, temperature_steps, equilibration_steps, number_of_angle_states, magnetic_field):
		#get the relevant attributes from the SpinLattice to have instances in the scope of temperature_sweep
		s_max = self.s_max
		theta = self.theta
		phi = self.phi
		energy = self.energy
		energy_calc = self.energy_calc
		negative_of_energy_calc = self.negative_of_energy_calc
		energy_calc_simple = self.energy_calc_simple
		super_exchange_field_calc = self.super_exchange_field_calc
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		single_ion_anisotropy = self.single_ion_anisotropy
		random_ijk_list = self.random_ijk_list
		
		#write the header to the status file
		f = open(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_run_status"),'w')
		f.write(str('x='+str(self.iron_doping_level)+' L='+str(self.edge_length)+"\n"))
		f.write("temperature (K), elapsed time (s), energy per site (K)\n")
		f.close()
		
		#when looking at the 0 to pi theta values that will be thermalized after choosing a random phi value from knowing the minimum energy
		spaced_theta_values = np.arccos(1-2*np.linspace(0,1,number_of_angle_states))
		
		#initialize all of the lists
		temporary_nn_pair_corr_list_ac = []
		temporary_nn_pair_corr_list_b = []
		
		temporary_pair_corr_list_ac = []
		temporary_pair_corr_list_b = []
			
		temperature_sweep_array = np.linspace(temperature_max, temperature_min, temperature_steps)
		E_temperature_array = np.zeros((temperature_steps, equilibration_steps))
		
		nn_pair_corr_abs_abc_temperature_array = np.zeros((temperature_steps, equilibration_steps))
		nn_pair_corr_ac_temperature_array = np.zeros((temperature_steps, equilibration_steps))
		nn_pair_corr_b_temperature_array = np.zeros((temperature_steps, equilibration_steps))
		
		A_temperature_array = np.zeros((3,3,temperature_steps, equilibration_steps)) #the first "3" is for total, Fe, or Mn, and the second "3" is for the x,y,z components
		G_temperature_array = np.zeros((3,3,temperature_steps, equilibration_steps)) #the first "3" is for total, Fe, or Mn, and the second "3" is for the x,y,z components
		
		print("sweeping temperature...")
		for temperature_index, temperature in enumerate(temperature_sweep_array):
			print("\ntemperature=",temperature)
			for equilibration_index in range(equilibration_steps):
				print("  i=", equilibration_index, end=':')
				for ijk in random_ijk_list:
					#i want to not call the ijk every time when i'm getting spin positions and such
					i,j,k = ijk[0], ijk[1], ijk[2]
					s_max_ijk = s_max[i,j,k]
					single_ion_anisotropy_ijk = single_ion_anisotropy[i,j,k]
					
					#calculate the super_exchange_field for the site in the coordinate system of the lattice
					super_exchange_field_c = super_exchange_field_calc(i,j,k)
					
					#put in the lab magnetic field
					super_exchange_field_c = super_exchange_field_c + magnetic_field
					
					#calculate the minimum positions of theta and phi for the site in the lattice coordinate system
					theta_min_c, phi_min_c = spop.optimize.fmin(energy_calc, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(theta[i,j,k], phi[i,j,k]), args = (super_exchange_field_c,single_ion_anisotropy_ijk,s_max_ijk), disp=0)
					
					#move to the frame in which the minimum energy position is along the z-axis 
					super_exchange_field_r = np.dot(my_rot_mat(theta_min_c, phi_min_c), super_exchange_field_c)
					single_ion_anisotropy_ijk_r = np.dot(my_rot_mat(theta_min_c, phi_min_c), single_ion_anisotropy_ijk)
					phi_thermal_r = random.random()*2*np.pi #phi is completely random in the rotated frame
					
					#make a list of energies to choose from when applying the Boltzmann statistics
					E_r_list = energy_calc((spaced_theta_values, phi_thermal_r),super_exchange_field_r,single_ion_anisotropy_ijk_r,s_max_ijk)
					E_r_list = np.add(E_r_list, -np.min(E_r_list)) #the energies are always less than zero, so we add the negative of the minimum to avoid large numbers in Boltzmann factor
					
					#from the list of energies, apply Boltzmann statistics to get the probability of each angle, and normalize
					P_r_list = np.exp(-E_r_list/temperature)
					P_r_list = P_r_list/np.sum(P_r_list)
					
					#make a weighted probability choice of the theta_r value									
					theta_thermal_r = np.random.choice(spaced_theta_values, p=P_r_list)
					
					#change from spherical to cartesian coordinates in prepration for rotation back to the lattice frame
					s_x_thermal_r = s_max_ijk*np.sin(theta_thermal_r)*np.cos(phi_thermal_r)
					s_y_thermal_r = s_max_ijk*np.sin(theta_thermal_r)*np.sin(phi_thermal_r)
					s_z_thermal_r = s_max_ijk*np.cos(theta_thermal_r)									
					s_thermal_r = np.array([s_x_thermal_r, s_y_thermal_r, s_z_thermal_r])

					#take the thermalized value for the spin orientation and rotate into the lattice frame
					s_thermal_c = np.dot(my_rot_mat(-theta_min_c, -phi_min_c), s_thermal_r)
					s_x_thermal_c, s_y_thermal_c, s_z_thermal_c = s_thermal_c
					
					theta_thermal_c = np.arccos(s_z_thermal_c / np.sqrt(s_x_thermal_c**2 + s_y_thermal_c**2 +s_z_thermal_c**2))
					phi_thermal_c = -np.arctan2(s_y_thermal_c, s_x_thermal_c)
					energy_thermal_c = energy_calc((theta_thermal_c, phi_thermal_c),super_exchange_field_c,single_ion_anisotropy_ijk,s_max_ijk)
					
					#update the SpinLattice parameters for the given site with the thermalized values for that equilibration step
					phi[i,j,k] = phi_thermal_c
					theta[i,j,k] = theta_thermal_c
					s_x[i,j,k] = s_max_ijk*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
					s_y[i,j,k] = s_max_ijk*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
					s_z[i,j,k] = s_max_ijk*np.cos(theta[i,j,k])
					energy[i,j,k] = energy_calc((theta[i,j,k],phi[i,j,k]),super_exchange_field_c,single_ion_anisotropy_ijk,s_max_ijk)
							
				#store the energies for each equilibration step, indexed by temperature
				E_temperature_array[temperature_index, equilibration_index] = np.sum(energy)/edge_length**3 #((temperature_steps, equilibration_steps))
				
				#store the AF order parameters for each equilibration step, indexed by temperature
				temp_a_x, temp_a_y, temp_a_z, temp_mn_a_x, temp_mn_a_y, temp_mn_a_z, temp_fe_a_x, temp_fe_a_y, temp_fe_a_z = np.divide(self.a_type_order_parameter_calc(), edge_length**3)
				temp_g_x, temp_g_y, temp_g_z, temp_mn_g_x, temp_mn_g_y, temp_mn_g_z, temp_fe_g_x, temp_fe_g_y, temp_fe_g_z = np.divide(self.g_type_order_parameter_calc(), edge_length**3)
				A_temperature_array[0,0,temperature_index, equilibration_index], A_temperature_array[0,1,temperature_index, equilibration_index], A_temperature_array[0,2,temperature_index, equilibration_index], A_temperature_array[1,0,temperature_index, equilibration_index], A_temperature_array[1,1,temperature_index, equilibration_index], A_temperature_array[1,2,temperature_index, equilibration_index], A_temperature_array[2,0,temperature_index, equilibration_index], A_temperature_array[2,1,temperature_index, equilibration_index], A_temperature_array[2,2,temperature_index, equilibration_index] = temp_a_x, temp_a_y, temp_a_z, temp_mn_a_x, temp_mn_a_y, temp_mn_a_z, temp_fe_a_x, temp_fe_a_y, temp_fe_a_z
				G_temperature_array[0,0,temperature_index, equilibration_index], G_temperature_array[0,1,temperature_index, equilibration_index], G_temperature_array[0,2,temperature_index, equilibration_index], G_temperature_array[1,0,temperature_index, equilibration_index], G_temperature_array[1,1,temperature_index, equilibration_index], G_temperature_array[1,2,temperature_index, equilibration_index], G_temperature_array[2,0,temperature_index, equilibration_index], G_temperature_array[2,1,temperature_index, equilibration_index], G_temperature_array[2,2,temperature_index, equilibration_index] = temp_g_x, temp_g_y, temp_g_z, temp_mn_g_x, temp_mn_g_y, temp_mn_g_z, temp_fe_g_x, temp_fe_g_y, temp_fe_g_z
				
				#store the nearest neighbor correlations for each equilibration step, indexed by temperature
				temp_nn_pair_corr_var_abs_abc, temp_nn_pair_corr_var_ac, temp_nn_pair_corr_var_b = np.divide(self.nn_pair_corr_calc(),edge_length**3)
				nn_pair_corr_abs_abc_temperature_array[temperature_index, equilibration_index] = temp_nn_pair_corr_var_abs_abc
				nn_pair_corr_ac_temperature_array[temperature_index, equilibration_index] = temp_nn_pair_corr_var_ac
				nn_pair_corr_b_temperature_array[temperature_index, equilibration_index] = temp_nn_pair_corr_var_b
				
			#write to a status file for this temperature
			f = open(self.file_prefix+str(str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_run_status"),'a')
			f.write(str(temperature) + ", " + str(time() - start_time) + ", " + str(np.sum(energy)/edge_length**3)+"\n")
			f.close()
			
			"""
			#working on this part start
			#to get the correlation length using pair_corr_calc rather than nn_pair_corr_calc
			temp_pair_corr_var_ac, temp_pair_corr_var_b = self.pair_corr_calc()
			temp_pair_corr_var_ac = np.delete(temp_pair_corr_var_ac, 0)
			temp_pair_corr_var_b = np.delete(temp_pair_corr_var_b, 0)
			temporary_pair_corr_list_ac.append(temp_pair_corr_var_ac)
			temporary_pair_corr_list_b.append(temp_pair_corr_var_b)
			print("\ntemporary_pair_corr_list_ac",temporary_pair_corr_list_ac)
			print("\ntemporary_pair_corr_list_b",temporary_pair_corr_list_b)
			#working on this part end
			
			temporary_nn_pair_corr_list_ac.append(temp_nn_pair_corr_var_ac)
			temporary_nn_pair_corr_list_b.append(temp_nn_pair_corr_var_b)
			"""
			
			
			print('\nfinal energy=', np.sum(energy), 'pair corr ac then b',temp_nn_pair_corr_var_ac, temp_nn_pair_corr_var_b)
			np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_s_x"), s_x)
			np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_s_y"), s_y)
			np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_s_z"), s_z)
			
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_E_temperature_array"), E_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_A_temperature_array"), A_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_G_temperature_array"), G_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_nn_pair_corr_abs_abc_temperature_array"), nn_pair_corr_abs_abc_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_nn_pair_corr_ac_temperature_array"), nn_pair_corr_ac_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_nn_pair_corr_b_temperature_array"), nn_pair_corr_b_temperature_array)
		np.save(str(self.file_prefix+str(int(start_time))+'_x='+str(self.iron_doping_level)+'_L='+str(self.edge_length) + "_temperature_sweep_array"), temperature_sweep_array)

			
		

		
	def random_ijk_list_generator(self):
		random_ijk_list = self.random_ijk_list
		edge_length = self.edge_length
		#random_ijk_list = []
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					random_ijk_list.append(np.array([i,j,k]))
		random.shuffle(random_ijk_list)


	def super_exchange_field_calc(self,i,j,k):
		#theta, phi = self.theta[i,j,k], self.phi[i,j,k]
		atom_type = self.atom_type
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		superexchange = self.superexchange_array[i,j,k]
		super_exchange_field_ijk = np.array([0,0,0])
		
		if i < edge_length-1:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[0]*np.array([s_x[i+1,j,k] , s_y[i+1,j,k] , s_z[i+1,j,k]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[0]*np.array([s_x[0,j,k] , s_y[0,j,k] , s_z[0,j,k]])
		if i > 0:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[1]*np.array([s_x[i-1,j,k] , s_y[i-1,j,k] , s_z[i-1,j,k]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[1]*np.array([s_x[edge_length-1,j,k] , s_y[edge_length-1,j,k] , s_z[edge_length-1,j,k]])
			
		if j < edge_length-1:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[2]*np.array([s_x[i,j+1,k] , s_y[i,j+1,k] , s_z[i,j+1,k]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[2]*np.array([s_x[i,0,k] , s_y[i,0,k] , s_z[i,0,k]])
		if j > 0:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[3]*np.array([s_x[i,j-1,k] , s_y[i,j-1,k] , s_z[i,j-1,k]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[3]*np.array([s_x[i,edge_length-1,k] , s_y[i,edge_length-1,k] , s_z[i,edge_length-1,k]])
			
		if k < edge_length-1:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[4]*np.array([s_x[i,j,k+1] , s_y[i,j,k+1] , s_z[i,j,k+1]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[4]*np.array([s_x[i,j,0] , s_y[i,j,0] , s_z[i,j,0]])
		if k > 0:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[5]*np.array([s_x[i,j,k-1] , s_y[i,j,k-1] , s_z[i,j,k-1]])
		else:
			super_exchange_field_ijk = super_exchange_field_ijk + superexchange[5]*np.array([s_x[i,j,edge_length-1] , s_y[i,j,edge_length-1] , s_z[i,j,edge_length-1]])
		return super_exchange_field_ijk

	def nn_pair_corr_calc(self):
		"""
		x, y, z is the direction of the spin
		a, b, c is the direction within the lattice
		"""
		edge_length = self.edge_length
	
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		
		nn_pair_corrxa = np.zeros(np.shape(s_x))
		nn_pair_corrya = np.zeros(np.shape(s_x))
		nn_pair_corrza = np.zeros(np.shape(s_x))
		
		nn_pair_corrxb = np.zeros(np.shape(s_x))
		nn_pair_corryb = np.zeros(np.shape(s_x))
		nn_pair_corrzb = np.zeros(np.shape(s_x))
		
		nn_pair_corrxc = np.zeros(np.shape(s_x))
		nn_pair_corryc = np.zeros(np.shape(s_x))
		nn_pair_corrzc = np.zeros(np.shape(s_x))
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					nn_pair_corrxa[i,j,k], nn_pair_corrya[i,j,k], nn_pair_corrza[i,j,k], nn_pair_corrxb[i,j,k], nn_pair_corryb[i,j,k], nn_pair_corrzb[i,j,k], nn_pair_corrxc[i,j,k], nn_pair_corryc[i,j,k], nn_pair_corrzc[i,j,k] = self.nn_pair_corr(i,j,k)
		abs_abc = np.sum(np.abs(nn_pair_corrxa)) + np.sum(np.abs(nn_pair_corrya)) + np.sum(np.abs(nn_pair_corrza)) + np.sum(np.abs(nn_pair_corrxb)) + np.sum(np.abs(nn_pair_corryb)) + np.sum(np.abs(nn_pair_corrzb)) + np.sum(np.abs(nn_pair_corrxc)) + np.sum(np.abs(nn_pair_corryc)) + np.sum(np.abs(nn_pair_corrzc))
		ac = np.sum(nn_pair_corrxa)+np.sum(nn_pair_corrya)+np.sum(nn_pair_corrza)  +  np.sum(nn_pair_corrxc)+np.sum(nn_pair_corryc)+np.sum(nn_pair_corrzc)
		b = np.sum(nn_pair_corrxb)+np.sum(nn_pair_corryb)+np.sum(nn_pair_corrzb)
		return abs_abc, ac ,b
			
	def nn_pair_corr(self,i,j,k):
		edge_length = self.edge_length
		s_max = self.s_max
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		if i < edge_length-1:
			nn_pair_corrxa_ijk = s_x[i,j,k]*s_x[i+1,j,k]/(s_max[i,j,k]*s_max[i+1,j,k])
			nn_pair_corrya_ijk = s_y[i,j,k]*s_y[i+1,j,k]/(s_max[i,j,k]*s_max[i+1,j,k])
			nn_pair_corrza_ijk = s_z[i,j,k]*s_z[i+1,j,k]/(s_max[i,j,k]*s_max[i+1,j,k])
		else:
			nn_pair_corrxa_ijk = s_x[i,j,k]*s_x[0,j,k]/(s_max[i,j,k]*s_max[0,j,k])
			nn_pair_corrya_ijk = s_y[i,j,k]*s_y[0,j,k]/(s_max[i,j,k]*s_max[0,j,k])
			nn_pair_corrza_ijk = s_z[i,j,k]*s_z[0,j,k]/(s_max[i,j,k]*s_max[0,j,k])
			
		if j < edge_length-1:
			nn_pair_corrxb_ijk = s_x[i,j,k]*s_x[i,j+1,k]/(s_max[i,j,k]*s_max[i,j+1,k])
			nn_pair_corryb_ijk = s_y[i,j,k]*s_y[i,j+1,k]/(s_max[i,j,k]*s_max[i,j+1,k])
			nn_pair_corrzb_ijk = s_z[i,j,k]*s_z[i,j+1,k]/(s_max[i,j,k]*s_max[i,j+1,k])
		else:
			nn_pair_corrxb_ijk = s_x[i,j,k]*s_x[i,0,k]/(s_max[i,j,k]*s_max[i,0,k])
			nn_pair_corryb_ijk = s_y[i,j,k]*s_y[i,0,k]/(s_max[i,j,k]*s_max[i,0,k])
			nn_pair_corrzb_ijk = s_z[i,j,k]*s_z[i,0,k]/(s_max[i,j,k]*s_max[i,0,k])
			
		if k < edge_length-1:
			nn_pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,k+1]/(s_max[i,j,k]*s_max[i,j,k+1])
			nn_pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,k+1]/(s_max[i,j,k]*s_max[i,j,k+1])
			nn_pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,k+1]/(s_max[i,j,k]*s_max[i,j,k+1])
		else:
			nn_pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,0]/(s_max[i,j,k]*s_max[i,j,0])
			nn_pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,0]/(s_max[i,j,k]*s_max[i,j,0])
			nn_pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,0]/(s_max[i,j,k]*s_max[i,j,0])

		return nn_pair_corrxa_ijk, nn_pair_corrya_ijk, nn_pair_corrza_ijk, nn_pair_corrxb_ijk, nn_pair_corryb_ijk, nn_pair_corrzb_ijk, nn_pair_corrxc_ijk, nn_pair_corryc_ijk, nn_pair_corrzc_ijk

		
	def pair_corr_calc(self):
		"""
		x, y, z is the direction of the spin
		a, b, c is the direction within the lattice
		"""
		edge_length = self.edge_length
	
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		
		
		
		pair_corrxa_sum = np.zeros(edge_length/2+1)
		pair_corrya_sum = np.zeros(edge_length/2+1)
		pair_corrza_sum = np.zeros(edge_length/2+1)
		
		pair_corrxb_sum = np.zeros(edge_length/2+1)
		pair_corryb_sum = np.zeros(edge_length/2+1)
		pair_corrzb_sum = np.zeros(edge_length/2+1)
		
		pair_corrxc_sum = np.zeros(edge_length/2+1)
		pair_corryc_sum = np.zeros(edge_length/2+1)
		pair_corrzc_sum = np.zeros(edge_length/2+1)
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					#self.pair_corr(i,j,k)
					pair_corrxa, pair_corrya, pair_corrza, pair_corrxb, pair_corryb, pair_corrzb, pair_corrxc, pair_corryc, pair_corrzc = self.pair_corr(i,j,k)
					
					pair_corrxa_sum = pair_corrxa_sum + pair_corrxa
					pair_corrya_sum = pair_corrya_sum + pair_corrya
					pair_corrza_sum = pair_corrza_sum + pair_corrza
					
					pair_corrxb_sum = pair_corrxb_sum + pair_corrxb
					pair_corryb_sum = pair_corryb_sum + pair_corryb
					pair_corrzb_sum = pair_corrzb_sum + pair_corrzb
					
					pair_corrxc_sum = pair_corrxc_sum + pair_corrxc
					pair_corryc_sum = pair_corryc_sum + pair_corryc
					pair_corrzc_sum = pair_corrzc_sum + pair_corrzc
		
		pair_corr_ac = pair_corrxa_sum + pair_corrya_sum + pair_corrza_sum  +  pair_corrxc_sum + pair_corryc_sum + pair_corrzc_sum
		pair_corr_b = pair_corrxb_sum + pair_corryb_sum + pair_corrzb_sum
		return pair_corr_ac, pair_corr_b

		
	def pair_corr(self,i,j,k):
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		#a,b,c are the lattice directions
		#x,y,z are the spin directions
		
		#moving along the a_axis
		pair_corrxa_ijk = np.zeros(edge_length/2+1)
		pair_corrya_ijk = np.zeros(edge_length/2+1)
		pair_corrza_ijk = np.zeros(edge_length/2+1)
		for i_var in range(0,edge_length):
			r_i = i - i_var
			if r_i > 0:
				if r_i > edge_length/2:
					r_i = r_i - edge_length/2
				#print("edge_length/2, r_i", edge_length/2, r_i)
				pair_corrxa_ijk[r_i] = pair_corrxa_ijk[r_i] + np.abs(s_x[i,j,k]*s_x[i_var,j,k])
				pair_corrya_ijk[r_i] = pair_corrya_ijk[r_i] + np.abs(s_y[i,j,k]*s_y[i_var,j,k])
				pair_corrza_ijk[r_i] = pair_corrza_ijk[r_i] + np.abs(s_z[i,j,k]*s_z[i_var,j,k])

		#moving along the b_axis
		pair_corrxb_ijk = np.zeros(edge_length/2+1)
		pair_corryb_ijk = np.zeros(edge_length/2+1)
		pair_corrzb_ijk = np.zeros(edge_length/2+1)
		for j_var in range(0,edge_length):
			r_j = j - j_var
			if r_j > 0:
				if r_j > edge_length/2:
					r_j = r_j - edge_length/2
				pair_corrxb_ijk[r_j] = pair_corrxb_ijk[r_j] + np.abs(s_x[i,j,k]*s_x[i,j_var,k])
				pair_corryb_ijk[r_j] = pair_corryb_ijk[r_j] + np.abs(s_y[i,j,k]*s_y[i,j_var,k])
				pair_corrzb_ijk[r_j] = pair_corrzb_ijk[r_j] + np.abs(s_z[i,j,k]*s_z[i,j_var,k])

		#moving along the c_axis
		pair_corrxc_ijk = np.zeros(edge_length/2+1)
		pair_corryc_ijk = np.zeros(edge_length/2+1)
		pair_corrzc_ijk = np.zeros(edge_length/2+1)
		for k_var in range(0,edge_length):
			r_k = k - k_var
			if r_k > 0:
				if r_k > edge_length/2:
					r_k = r_k - edge_length/2
				pair_corrxc_ijk[r_k] = pair_corrxc_ijk[r_k] + np.abs(s_x[i,j,k]*s_x[i,j,k_var])
				pair_corryc_ijk[r_k] = pair_corryc_ijk[r_k] + np.abs(s_y[i,j,k]*s_y[i,j,k_var])
				pair_corrzc_ijk[r_k] = pair_corrzc_ijk[r_k] + np.abs(s_z[i,j,k]*s_z[i,j,k_var])
				
		return pair_corrxa_ijk, pair_corrya_ijk, pair_corrza_ijk, pair_corrxb_ijk, pair_corryb_ijk, pair_corrzb_ijk, pair_corrxc_ijk, pair_corryc_ijk, pair_corrzc_ijk

		

	def make_op_masks(self):
		atom_type = self.atom_type

		#mn_g_type_mask = self.mn_g_type_mask
		#mn_a_type_mask = self.mn_a_type_mask
		#fe_g_type_mask = self.fe_g_type_mask
		#fe_a_type_mask = self.fe_a_type_mask
		
		g_type_mask = self.g_type_mask
		a_type_mask = self.a_type_mask
		edge_length = self.edge_length
		a_type_mask[::,::2,::] = 1
		a_type_mask[::,1::2,::] = -1

		for i in range(edge_length):
			for j in range(edge_length):
				for k in range(edge_length):
					if np.mod(i+j+k,2) == 0:
						g_type_mask[i,j,k] = 1
					else:
						g_type_mask[i,j,k] = -1
						
		self.mn_g_type_mask = np.multiply(atom_type, g_type_mask)
		self.mn_a_type_mask = np.multiply(atom_type, a_type_mask)
		self.fe_g_type_mask = np.multiply(1-atom_type, g_type_mask)
		self.fe_a_type_mask = np.multiply(1-atom_type, a_type_mask)
		
		#print('mn_g_type_mask', mn_g_type_mask)
		#print('self.mn_g_type_mask', self.mn_g_type_mask)
		#exit()
				
	def a_type_order_parameter_calc(self):
		edge_length = self.edge_length
		a_type_mask = self.a_type_mask
		mn_a_type_mask = self.mn_a_type_mask
		fe_a_type_mask = self.fe_a_type_mask

		a_x = np.sum(np.multiply(a_type_mask, self.s_x))
		a_y = np.sum(np.multiply(a_type_mask, self.s_y))
		a_z = np.sum(np.multiply(a_type_mask, self.s_z))
		
		mn_a_x = np.sum(np.multiply(mn_a_type_mask, self.s_x))
		mn_a_y = np.sum(np.multiply(mn_a_type_mask, self.s_y))
		mn_a_z = np.sum(np.multiply(mn_a_type_mask, self.s_z))
		
		fe_a_x = np.sum(np.multiply(fe_a_type_mask, self.s_x))
		fe_a_y = np.sum(np.multiply(fe_a_type_mask, self.s_y))
		fe_a_z = np.sum(np.multiply(fe_a_type_mask, self.s_z))
		
		#print(mn_a_type_mask)
		#print(a_type_mask)
		
		#print(a_x, a_y, a_z, mn_a_x, mn_a_y, mn_a_z)

		return a_x, a_y, a_z, mn_a_x, mn_a_y, mn_a_z, fe_a_x, fe_a_y, fe_a_z

	def g_type_order_parameter_calc(self):
		edge_length = self.edge_length
		g_type_mask = self.g_type_mask
		mn_g_type_mask = self.mn_g_type_mask
		fe_g_type_mask = self.fe_g_type_mask

		g_x = np.sum(np.multiply(g_type_mask, self.s_x))
		g_y = np.sum(np.multiply(g_type_mask, self.s_y))
		g_z = np.sum(np.multiply(g_type_mask, self.s_z))
		
		mn_g_x = np.sum(np.multiply(mn_g_type_mask, self.s_x))
		mn_g_y = np.sum(np.multiply(mn_g_type_mask, self.s_y))
		mn_g_z = np.sum(np.multiply(mn_g_type_mask, self.s_z))
		
		fe_g_x = np.sum(np.multiply(fe_g_type_mask, self.s_x))
		fe_g_y = np.sum(np.multiply(fe_g_type_mask, self.s_y))
		fe_g_z = np.sum(np.multiply(fe_g_type_mask, self.s_z))

		return g_x, g_y, g_z, mn_g_x, mn_g_y, mn_g_z, fe_g_x, fe_g_y, fe_g_z

	def bond_list_calc(self):
		atom_type = self.atom_type
		edge_length = self.edge_length
		number_of_mn_mn = 0
		number_of_fe_fe = 0
		number_of_mn_fe = 0
		number_of_fe_mn = 0
		#print(atom_type)
		for i in range(edge_length):
			for j in range(edge_length):
				for k in range(edge_length):
					#print(atom_type[i,j,k])
					if 1:#atom_type[i,j,k]:
						if i < edge_length-1:
							bond_identifier_1 = atom_type[i,j,k] + atom_type[i+1,j,k]
						else:
							bond_identifier_1 = atom_type[i,j,k] + atom_type[0,j,k]
						if i > 0:
							bond_identifier_2 = atom_type[i,j,k] + atom_type[i-1,j,k]
						else:
							bond_identifier_2 = atom_type[i,j,k] + atom_type[edge_length-1,j,k]
						if j < edge_length-1:
							bond_identifier_3 = atom_type[i,j,k] + atom_type[i,j+1,k]
						else:
							bond_identifier_3 = atom_type[i,j,k] + atom_type[i,0,k]
						if j > 0:
							bond_identifier_4 = atom_type[i,j,k] + atom_type[i,j-1,k]
						else:
							bond_identifier_4 = atom_type[i,j,k] + atom_type[i,edge_length-1,k]
						if k < edge_length-1:
							bond_identifier_5 = atom_type[i,j,k] + atom_type[i,j,k+1]
						else:
							bond_identifier_5 = atom_type[i,j,k] + atom_type[i,j,0]
						if k > 0:
							bond_identifier_6 = atom_type[i,j,k] + atom_type[i,j,k-1]
						else:
							bond_identifier_6 = atom_type[i,j,k] + atom_type[i,j,edge_length-1]
					else:
						print('Fe')
					bond_identifier = [bond_identifier_1, bond_identifier_2, bond_identifier_3, bond_identifier_4, bond_identifier_5, bond_identifier_6]
					#print(bond_identifier)
					for bond_identifier_i in bond_identifier:
						if bond_identifier_i == 2:
							number_of_mn_mn = number_of_mn_mn + 1
						elif bond_identifier_i == 0:
							number_of_fe_fe = number_of_fe_fe + 1
						else:
							number_of_mn_fe = number_of_mn_fe + 1
		number_of_mn_mn, number_of_fe_fe, number_of_mn_fe = number_of_mn_mn/2.0, number_of_fe_fe/2.0, number_of_mn_fe/2.0
		print(number_of_mn_mn, number_of_fe_fe, number_of_mn_fe)
class PairCorrelation(object):
	""" this class is defined to house the pair correlations
	nn_pair_corrxa, nn_pair_corrya, nn_pair_corrza,
	nn_pair_corrxb, nn_pair_corryb, nn_pair_corrzb,
	nn_pair_corrxc, nn_pair_corryc, nn_pair_corrzc
	"""
	
def my_rot_mat(theta,phi):
	"""
	this rotation matrix will rotate a coordinate system such that the point at (theta,phi) goes to [0,0,1]
	"""
	c = np.cos
	s = np.sin
	t = theta
	p = phi
	
	R11 = s(p)**2 + c(p)**2 * c(t)#c(t) + s(p)**2*(1+c(t))
	R12 = -s(p)*c(p)*(1-c(t))
	R13 = -c(p)*s(t)
	
	R21 = -s(p)*c(p)*(1-c(t))#c(p)*s(p)*(1-c(t))
	R22 = c(p)**2 + s(p)**2 * c(t) #c(t) + c(p)**2*(1+c(t))
	R23 = -s(p)*s(t)
	
	R31 = c(p)*s(t)
	R32 = s(p)*s(t)
	R33 = c(t)
	
	R = np.array([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
	
	return R

	
def nn_pair_corr(i,j,k):

    if i < edge_length-1:
        nn_pair_corrxa_ijk = s_x[i,j,k]*s_x[i+1,j,k]
        nn_pair_corrya_ijk = s_y[i,j,k]*s_y[i+1,j,k]
        nn_pair_corrza_ijk = s_z[i,j,k]*s_z[i+1,j,k]
    else:
        nn_pair_corrxa_ijk = s_x[i,j,k]*s_x[0,j,k]
        nn_pair_corrya_ijk = s_y[i,j,k]*s_y[0,j,k]
        nn_pair_corrza_ijk = s_z[i,j,k]*s_z[0,j,k]
        
    if j < edge_length-1:
        nn_pair_corrxb_ijk = s_x[i,j,k]*s_x[i,j+1,k]
        nn_pair_corryb_ijk = s_y[i,j,k]*s_y[i,j+1,k]
        nn_pair_corrzb_ijk = s_z[i,j,k]*s_z[i,j+1,k]
    else:
        nn_pair_corrxb_ijk = s_x[i,j,k]*s_x[i,0,k]
        nn_pair_corryb_ijk = s_y[i,j,k]*s_y[i,0,k]
        nn_pair_corrzb_ijk = s_z[i,j,k]*s_z[i,0,k]
        
    if k < edge_length-1:
        nn_pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,k+1]
        nn_pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,k+1]
        nn_pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,k+1]
    else:
        nn_pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,0]
        nn_pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,0]
        nn_pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,0]

    return nn_pair_corrxa_ijk, nn_pair_corrya_ijk, nn_pair_corrza_ijk, nn_pair_corrxb_ijk, nn_pair_corryb_ijk, nn_pair_corrzb_ijk, nn_pair_corrxc_ijk, nn_pair_corryc_ijk, nn_pair_corrzc_ijk







	
	
