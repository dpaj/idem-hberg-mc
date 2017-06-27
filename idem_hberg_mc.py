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


hbar = 6.626070040e-34 #SI units

#arbitrary number visualize spins
moment_visualization_scale_factor = 0.1

edge_length = 4 #length of 3-dimensional lattice, such that N = edge_length^3
s_max = 2 #spin number
single_ion_anisotropy = 1 #anisotropy value, in Kelvin units
superexchange = 1 #nearest neighbor isotropic superexchange parameter, in Kelvin units
magnetic_field = np.array([0,0,0]) #magnetic field, in Kelvin units
t_debye = 6*2*superexchange*s_max*s_max

E0_single_spin = -np.abs(6*superexchange*s_max**2)-single_ion_anisotropy*s_max**2

class SpinLattice(object):
	""" this class is defined to house the lattice parameters and variables
	
	s_x, s_y, s_z = these are edge_length**3 arrays of the cartesian components of the spins
	phi = this is an edge_length**3 array of the azimuthal angles, each of which has a range of [0,2*pi]
	theta = this is an edge_length**3 array of the elevation angle, each of which has a range of [0,pi]
	energy = this is an edge_length**3 array of the site energies

	"""
	def __init__(self, edge_length=None, s_max=None, single_ion_anisotropy=None, superexchange=None, magnetic_field=None, s_x=None, s_y=None, s_z=None, phi=None, theta=None, energy=None, total_energy=None, random_ijk_array=None, possible_angles_list=None):
		self.edge_length = edge_length
		self.single_ion_anisotropy = single_ion_anisotropy
		self.s_max = s_max
		self.superexchange = superexchange
		self.magnetic_field = magnetic_field
		self.s_x = np.zeros((edge_length,edge_length,edge_length))
		self.s_y = np.zeros((edge_length,edge_length,edge_length))
		self.s_z = np.zeros((edge_length,edge_length,edge_length))
		self.phi = np.zeros((edge_length,edge_length,edge_length))
		self.theta = np.zeros((edge_length,edge_length,edge_length))
		self.energy = np.zeros((edge_length,edge_length,edge_length))
		self.total_energy = 0
		self.random_ijk_list = []
		self.possible_angles_list = []
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
	def temp_energy_calc(self, x, i, j, k, energy_min, theta_min, phi_min, s_x_min, s_y_min, s_z_min, temperature):
		theta, phi = x[0], x[1]
		#print(theta)
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		s_max = self.s_max
		
		angle_between_vectors = np.arccos(  (  (s_x[i,j,k]*s_x_min) + (s_y[i,j,k]*s_y_min) + (s_z[i,j,k]*s_z_min)  ) / s_max**2  )
		#print(angle_between_vectors)
		return (angle_between_vectors-temperature/10.0*np.pi*0.5)**2 + (self.energy_calc(x,i,j,k)-energy_min-temperature)**2
	def energy_calc(self, x, i, j, k):
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		theta, phi = x[0], x[1]
		s_x_ijk = s_max*np.sin(theta)*np.cos(phi)
		s_y_ijk = s_max*np.sin(theta)*np.sin(phi)
		s_z_ijk = s_max*np.cos(theta)
		energy_ijk = 0

		energy_ijk += -single_ion_anisotropy*s_x_ijk**2

		#if Type[i,j,k]== 1:
		#    energy_ijk += DMn[0]*s_x_ijk**2 + DMn[1]*s_y_ijk**2 + DMn[2]*s_z_ijk**2
		#else:
		#    energy_ijk += DFe[0]*s_x_ijk**2 + DFe[1]*s_y_ijk**2 + DFe[2]*s_z_ijk**2
		#    #energy_ijk += DFe_cubic*(s_x_ijk**2*s_y_ijk**2+s_y_ijk**2*s_z_ijk**2+s_z_ijk**2*s_x_ijk**2)
		if i < edge_length-1:
			energy_ijk += superexchange*(s_x_ijk*s_x[i+1,j,k] + s_y_ijk*s_y[i+1,j,k] + s_z_ijk*s_z[i+1,j,k])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[0,j,k] + s_y_ijk*s_y[0,j,k] + s_z_ijk*s_z[0,j,k])
		if i > 0:
			energy_ijk += superexchange*(s_x_ijk*s_x[i-1,j,k] + s_y_ijk*s_y[i-1,j,k] + s_z_ijk*s_z[i-1,j,k])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[edge_length-1,j,k] + s_y_ijk*s_y[edge_length-1,j,k] + s_z_ijk*s_z[edge_length-1,j,k])
			
		if j < edge_length-1:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j+1,k] + s_y_ijk*s_y[i,j+1,k] + s_z_ijk*s_z[i,j+1,k])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,0,k] + s_y_ijk*s_y[i,0,k] + s_z_ijk*s_z[i,0,k])
		if j > 0:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j-1,k] + s_y_ijk*s_y[i,j-1,k] + s_z_ijk*s_z[i,j-1,k])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,edge_length-1,k] + s_y_ijk*s_y[i,edge_length-1,k] + s_z_ijk*s_z[i,edge_length-1,k])
			
		if k < edge_length-1:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j,k+1] + s_y_ijk*s_y[i,j,k+1] + s_z_ijk*s_z[i,j,k+1])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j,0] + s_y_ijk*s_y[i,j,0] + s_z_ijk*s_z[i,j,0])
		if k > 0:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j,k-1] + s_y_ijk*s_y[i,j,k-1] + s_z_ijk*s_z[i,j,k-1])
		else:
			energy_ijk += superexchange*(s_x_ijk*s_x[i,j,edge_length-1] + s_y_ijk*s_y[i,j,edge_length-1] + s_z_ijk*s_z[i,j,edge_length-1])
		return energy_ijk
	def init_rand_arrays(self):
		edge_length, s_x, s_y, s_z, phi, theta, energy = self.edge_length, self.s_x, self.s_y, self.s_z, self.phi, self.theta, self.energy
		#initialize the spin momentum vectors to have a random direction
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					phi[i,j,k] = np.random.rand()*2*np.pi
					theta[i,j,k] = np.arccos(1.0-2.0*np.random.rand())# asdf
					s_x[i,j,k] = s_max*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
					s_y[i,j,k] = s_max*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
					s_z[i,j,k] = s_max*np.cos(theta[i,j,k])
					energy[i,j,k] = self.energy_calc((theta[i,j,k],phi[i,j,k]),i,j,k)

		return s_x, s_y, s_z, phi, theta, energy
	def total_energy_calc(self):
		self.total_energy = np.sum(self.energy)
		return total_energy
	def temperature_sweep(self, temperature_max, temperature_min, temperature_steps, equilibration_steps, number_of_angle_states):
		theta = self.theta
		phi = self.phi
		energy = self.energy
		energy_calc = self.energy_calc
		edge_length = self.edge_length
		s_x = self.s_x
		s_y = self.s_y
		s_z = self.s_z
		random_ijk_list = self.random_ijk_list
		possible_angles_list = self.possible_angles_list
		temp_energy_calc = self.temp_energy_calc
		
		print("sweeping temperature...")
		for temperature in np.linspace(temperature_max, temperature_min, temperature_steps):
			print("\ntemperature=",temperature)
			for equilibration_index in np.linspace(0,equilibration_steps-1,equilibration_steps):
				print("  equilibration index=", equilibration_index, end=':')
				for ijk in random_ijk_list:
					i,j,k = ijk[0], ijk[1], ijk[2]
					old_energy = energy[i,j,k]*1.0
					old_phi = phi[i,j,k]*1.0
					old_theta = theta[i,j,k]*1.0
					old_s_x = s_x[i,j,k]*1.0
					old_s_y = s_y[i,j,k]*1.0
					old_s_z = s_z[i,j,k]*1.0
					
					theta_min, phi_min = spop.optimize.fmin(energy_calc, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(theta[i,j,k],phi[i,j,k]), args = (i,j,k), disp=0)
					
					s_x_min = s_max*np.sin(theta_min)*np.cos(phi_min)
					s_y_min = s_max*np.sin(theta_min)*np.sin(phi_min)
					s_z_min = s_max*np.cos(theta_min)
					energy_min = energy_calc((theta_min, phi_min),i,j,k)
					
					theta_min, phi_min = spop.optimize.fmin(temp_energy_calc, \
					maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(theta[i,j,k],phi[i,j,k]), \
					args = (i,j,k, energy_min, theta_min, phi_min, s_x_min, s_y_min, s_z_min, temperature), disp=0)

					"""
					temp_e_array = np.zeros(len(possible_angles_list))
					for index, angle_state in enumerate(possible_angles_list):
						temp_e_array[index] = energy_calc((angle_state[0], angle_state[1]),i,j,k)
					my_energy_min_index = np.argmin(temp_e_array)
					my_energy_min = temp_e_array[my_energy_min_index]
					
					my_energy_max_index = np.argmax(temp_e_array)
					my_energy_max = temp_e_array[my_energy_max_index]
					my_energy_range = my_energy_max - my_energy_min
					print('my_energy_max', my_energy_max, 'my_energy_max_index energy', temp_e_array[my_energy_max_index])
					print('my_energy_range', my_energy_range)
					
					zeroed_e_array = temp_e_array - my_energy_min
					#plt.plot(zeroed_e_array,'o')
					#plt.show()
					partition_function = np.exp(-(zeroed_e_array/temperature))
					print(np.sum(partition_function))
					#print(energy_min, np.min(temp_e_array))
					
					"""
					
					
					phi[i,j,k] = phi_min
					theta[i,j,k] = theta_min
					s_x[i,j,k] = s_max*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
					s_y[i,j,k] = s_max*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
					s_z[i,j,k] = s_max*np.cos(theta[i,j,k])
					energy[i,j,k] = energy_calc((theta[i,j,k],phi[i,j,k]),i,j,k)
					
				print(np.sum(energy), end=', ')
					
	def random_ijk_list_generator(self):
		random_ijk_list = self.random_ijk_list
		edge_length = self.edge_length
		#random_ijk_list = []
		for i in range(0,edge_length):
			for j in range(0,edge_length):
				for k in range(0,edge_length):
					random_ijk_list.append(np.array([i,j,k]))
		random.shuffle(random_ijk_list)
	def possible_angles_list_generator(self, N):
		possible_angles_list = self.possible_angles_list
		phi = np.zeros(N)
		theta = np.zeros(N)
		for k in range(1,N):
			h = -1.0 +2.0*(k-1.0)/(N-1.0)
			#print(h)
			theta[k] = np.arccos(h)
			if k == 1 or k == N:
				phi[k] = 0
			else:
				phi[k] = (phi[k-1] + 3.6/np.sqrt(N*(1.0-h**2))) % (2.0*np.pi)
			possible_angles_list.append(np.array([theta[k], phi[k]]))
			
			

							
class PairCorrelation(object):
	""" this class is defined to house the pair correlations
	pair_corrxa, pair_corrya, pair_corrza,
	pair_corrxb, pair_corryb, pair_corrzb,
	pair_corrxc, pair_corryc, pair_corrzc
	"""
	
	
def pair_corr(i,j,k):

    if i < edge_length-1:
        pair_corrxa_ijk = s_x[i,j,k]*s_x[i+1,j,k]
        pair_corrya_ijk = s_y[i,j,k]*s_y[i+1,j,k]
        pair_corrza_ijk = s_z[i,j,k]*s_z[i+1,j,k]
    else:
        pair_corrxa_ijk = s_x[i,j,k]*s_x[0,j,k]
        pair_corrya_ijk = s_y[i,j,k]*s_y[0,j,k]
        pair_corrza_ijk = s_z[i,j,k]*s_z[0,j,k]
        
    if j < edge_length-1:
        pair_corrxb_ijk = s_x[i,j,k]*s_x[i,j+1,k]
        pair_corryb_ijk = s_y[i,j,k]*s_y[i,j+1,k]
        pair_corrzb_ijk = s_z[i,j,k]*s_z[i,j+1,k]
    else:
        pair_corrxb_ijk = s_x[i,j,k]*s_x[i,0,k]
        pair_corryb_ijk = s_y[i,j,k]*s_y[i,0,k]
        pair_corrzb_ijk = s_z[i,j,k]*s_z[i,0,k]
        
    if k < edge_length-1:
        pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,k+1]
        pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,k+1]
        pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,k+1]
    else:
        pair_corrxc_ijk = s_x[i,j,k]*s_x[i,j,0]
        pair_corryc_ijk = s_y[i,j,k]*s_y[i,j,0]
        pair_corrzc_ijk = s_z[i,j,k]*s_z[i,j,0]

    return pair_corrxa_ijk, pair_corrya_ijk, pair_corrza_ijk, pair_corrxb_ijk, pair_corryb_ijk, pair_corrzb_ijk, pair_corrxc_ijk, pair_corryc_ijk, pair_corrzc_ijk

def local_field(x,i,j,k):
    theta, phi = x[0], x[1]
    s_x_ijk = s_max*np.sin(theta)*np.cos(phi)
    s_y_ijk = s_max*np.sin(theta)*np.sin(phi)
    s_z_ijk = s_max*np.cos(theta)
    local_field_ijk = np.array([0,0,0])
    
    if i < edge_length-1:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i+1,j,k] , s_y[i+1,j,k] , s_z[i+1,j,k]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[0,j,k] , s_y[0,j,k] , s_z[0,j,k]])
    if i > 0:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i-1,j,k] , s_y[i-1,j,k] , s_z[i-1,j,k]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[edge_length-1,j,k] , s_y[edge_length-1,j,k] , s_z[edge_length-1,j,k]])
        
    if j < edge_length-1:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j+1,k] , s_y[i,j+1,k] , s_z[i,j+1,k]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,0,k] , s_y[i,0,k] , s_z[i,0,k]])
    if j > 0:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j-1,k] , s_y[i,j-1,k] , s_z[i,j-1,k]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,edge_length-1,k] , s_y[i,edge_length-1,k] , s_z[i,edge_length-1,k]])
        
    if k < edge_length-1:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j,k+1] , s_y[i,j,k+1] , s_z[i,j,k+1]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j,0] , s_y[i,j,0] , s_z[i,j,0]])
    if k > 0:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j,k-1] , s_y[i,j,k-1] , s_z[i,j,k-1]])
    else:
        local_field_ijk = local_field_ijk + superexchange*np.array([s_x[i,j,edge_length-1] , s_y[i,j,edge_length-1] , s_z[i,j,edge_length-1]])
    return local_field_ijk

def non_min_energy_calc(x, non_min_e, i, j, k):
    return (energy_calc(x,i,j,k)-non_min_e)**2

def phonon_probability(E,T):
    return phonon_prefactor(E,T)*unscaled_phonon_probability(E,T)

def unscaled_phonon_probability(E,T):
    if E < t_debye:
        #return E**2/(np.exp(E/T)-1)
        #return E**8/(np.exp(E/T)-1)
        return np.exp(-(E-E0_single_spin)/T)
    else:
        return 0

def phonon_prefactor(E,T):
    ans, err = quad(unscaled_phonon_probability, 0, t_debye, args=(T))
    if err/ans < 1e-3:
        return 1/ans
    else:
        return "integration failed"

def phonon_energy_spectral_density(E,T):
    return E*phonon_probability(E,T)

def phonon_cdf(E,T):
    return quad(phonon_probability, 0, E, args=(T))



def createInvCDF(t):
    
    E = np.linspace(1e-6,t_debye,500)


    #state_probability = np.zeros(np.shape(E))
    #farfle = np.zeros(np.shape(E))
    cdf = np.zeros(np.shape(E))
    for i, e_ in enumerate(E):
        
        #state_probability[i] = phonon_probability(e_,t)
        #farfle[i] = phonon_energy_spectral_density(e_,t)
        cdf[i], err = phonon_cdf(e_,t) #quad(phonon_probability, 0, e_, args=(t))
        print(t, i, cdf[i],err)



    inv_cdf = interp1d(cdf, E)

    #plt.plot(cdf,E)
    #plt.figure()
    #plt.plot(E,cdf)
    #plt.figure()
    #plt.plot(E, inv_cdf(E))



    #r = np.random.uniform(0,t_debye, 1000)
    #ys = infv_cdf(r)

    #plt.plot(E,state_probability)
    #plt.plot(E,cdf)

    #plt.figure()

    #plt.plot(E,farfle)
    



    if 0:
        random_energy_list = []
        for i in range(0,10000):
            random_energy = inv_cdf(np.random.rand())
            while (random_energy > t_debye):
                random_energy = inv_cdf(np.random.rand())
            
            #unweighted_random_energy = np.random.rand()*t_debye
            #random_energy = phonon_energy_spectral_density(unweighted_random_energy,t)
            #print(phonon_energy_spectral_density(unweighted_random_energy,t))
            #random_energy = np.random.rand()*t_debye
            #print(random_energy)
            random_energy_list.append(random_energy)

        #random_energy_histogram, bin_edges = np.histogram(random_energy_list, bins=range(0,int(t_debye)))
        plt.figure()
        plt.hist(random_energy_list, bins = 50)
    return inv_cdf

	
	
print(time()-start_time)
	
my_lattice = SpinLattice(edge_length, s_max, single_ion_anisotropy, superexchange, magnetic_field)
my_lattice.init_rand_arrays()
print(time()-start_time)
print('start debugging')

print('end debugging')



my_lattice.random_ijk_list_generator()
my_lattice.possible_angles_list_generator(1000)
#print(my_lattice.possible_angles_list)
my_lattice.temperature_sweep(temperature_max=10.0, temperature_min=1.0, temperature_steps=2.0, equilibration_steps=20, number_of_angle_states=100)

print('\ntime=', time()-start_time)
exit()


	
t_list = np.linspace(10,2,20)
Emin_list = []

pcxa_list = []
pcya_list = []
pcza_list = []

pcxb_list = []
pcyb_list = []
pczb_list = []

pcxc_list = []
pcyc_list = []
pczc_list = []

for t in t_list:
    inv_cdf = createInvCDF(t)

    energy_list = []
    energy_list.append(np.sum(energy))
    #now do the energy minimization procedure
    for shmoo in range(0,10):
        for i in range(0,edge_length):
            for j in range(0,edge_length):
                for k in range(0,edge_length):
                    phonon_energy = inv_cdf(np.random.rand())
                    #print(local_field((theta[i,j,k],phi[i,j,k]),i,j,k))
                    newtheta, newphi = spop.optimize.fmin(energy_calc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(theta[i,j,k],phi[i,j,k]), args = (i,j,k), disp=0)
                    
                    
                    print(t, i,j,k,newtheta, newphi, energy_calc((newtheta, newphi),i,j,k), phonon_energy)
                    phi[i,j,k] = newphi
                    theta[i,j,k] = newtheta
                    s_x[i,j,k] = s_max*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
                    s_y[i,j,k] = s_max*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
                    s_z[i,j,k] = s_max*np.cos(theta[i,j,k])
                    energy[i,j,k] = energy_calc((theta[i,j,k],phi[i,j,k]),i,j,k)


                    non_min_e = energy[i,j,k]+phonon_energy
                    if non_min_e > 6*2*superexchange*s_max*s_max:
                        non_min_e = 6*2*superexchange*s_max*s_max
                    newtheta, newphi = spop.optimize.fmin(non_min_energy_calc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(np.arccos(1.0-2.0*np.random.rand()),np.random.rand()*np.pi), args = (non_min_e,i,j,k), disp=0)
                    phi[i,j,k] = newphi
                    theta[i,j,k] = newtheta
                    s_x[i,j,k] = s_max*np.sin(theta[i,j,k])*np.cos(phi[i,j,k])
                    s_y[i,j,k] = s_max*np.sin(theta[i,j,k])*np.sin(phi[i,j,k])
                    s_z[i,j,k] = s_max*np.cos(theta[i,j,k])
                    energy[i,j,k] = energy_calc((theta[i,j,k],phi[i,j,k]),i,j,k)                
                    #newtheta, newphi = spop.optimize.fmin(non_min_energy_calc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(theta[i,j,k],phi[i,j,k]), args = (phonon_energy,i,j,k), disp=0)
                    
        energy_list.append(np.sum(energy))

    Emin_list.append(np.min(energy_list))
    #print(energy_list)
    #print(-edge_length**3*2*superexchange*s_max*s_max*3, -edge_length**3*2*superexchange*s_max*s_max*3-edge_length**3*2*single_ion_anisotropy*s_max*s_max)
    #print(phonon_energy)
    if 0: #should each 3d map of the spins be drawn?
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #plot solution
        for i in range(0,edge_length):
            for j in range(0,edge_length):
                for k in range(0,edge_length):
                    ax.scatter(i, j, k, color = 'black', marker='o')
                    ax.plot([i,i+s_x[i,j,k]*moment_visualization_scale_factor], [j,j+s_y[i,j,k]*moment_visualization_scale_factor], [k,k+s_z[i,j,k]*moment_visualization_scale_factor], color = 'black')

    #calculate the pair correlations
    for i in range(0,edge_length):
        for j in range(0,edge_length):
            for k in range(0,edge_length):
                pair_corrxa[i,j,k], pair_corrya[i,j,k], pair_corrza[i,j,k], pair_corrxb[i,j,k], pair_corryb[i,j,k], pair_corrzb[i,j,k], pair_corrxc[i,j,k], pair_corryc[i,j,k], pair_corrzc[i,j,k] = pair_corr(i,j,k)


    print("np.sum(pair_corrxa), np.sum(pair_corrya), np.sum(pair_corrza)")
    print(np.sum(pair_corrxa), np.sum(pair_corrya), np.sum(pair_corrza))

    print("np.sum(pair_corrxb), np.sum(pair_corryb), np.sum(pair_corrzb)")
    print(np.sum(pair_corrxb), np.sum(pair_corryb), np.sum(pair_corrzb))

    print("np.sum(pair_corrxc), np.sum(pair_corryc), np.sum(pair_corrzc)")
    print(np.sum(pair_corrxc), np.sum(pair_corryc), np.sum(pair_corrzc))

    pcxa_list.append(np.sum(pair_corrxa))
    pcya_list.append(np.sum(pair_corrya))
    pcza_list.append(np.sum(pair_corrza))

    pcxb_list.append(np.sum(pair_corrxb))
    pcyb_list.append(np.sum(pair_corryb))
    pczb_list.append(np.sum(pair_corrzb))

    pcxc_list.append(np.sum(pair_corrxc))
    pcyc_list.append(np.sum(pair_corryc))
    pczc_list.append(np.sum(pair_corrzc))

end_time = time()

print("time in minutes =", -(start_time-end_time)/60.0)

plt.plot(t_list, pcxa_list, label = 'xa')
plt.plot(t_list, pcxb_list, label = 'xb')
plt.plot(t_list, pcxc_list, label = 'xc')

plt.plot(t_list, pcya_list, label = 'ya')
plt.plot(t_list, pcyb_list, label = 'yb')
plt.plot(t_list, pcyc_list, label = 'yc')

plt.plot(t_list, pcza_list, label = 'za')
plt.plot(t_list, pczb_list, label = 'zb')
plt.plot(t_list, pczc_list, label = 'zc')

plt.legend()
plt.show()
