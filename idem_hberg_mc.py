import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize as spop
from scipy.integrate import quad
from scipy.interpolate import interp1d

from time import time

start_time = time()

hbar = 6.626070040e-34 #SI units

moment_visualization_scale_factor = 0.1

L = 6
S = 2
D = 0
J = 1
h = np.array([0,0,0])
t_debye = 6*2*J*S*S #number of nearest neighbors, factor of 2 for spin flip, superexchange energy
#t_debye = 6*2*J*1*S #number of nearest neighbors, factor of 2 for spin flip, superexchange energy
#t_debye = 2*J*S
#t_debye = t_debye +2*D*S*S  # add anisotropy term to debye, really this depends on the direction, have to think about it...

def PairCorr(i,j,k):

    if i < L-1:
        PairCorrxa_ijk = Sx[i,j,k]*Sx[i+1,j,k]
        PairCorrya_ijk = Sy[i,j,k]*Sy[i+1,j,k]
        PairCorrza_ijk = Sz[i,j,k]*Sz[i+1,j,k]
    else:
        PairCorrxa_ijk = Sx[i,j,k]*Sx[0,j,k]
        PairCorrya_ijk = Sy[i,j,k]*Sy[0,j,k]
        PairCorrza_ijk = Sz[i,j,k]*Sz[0,j,k]
        
    if j < L-1:
        PairCorrxb_ijk = Sx[i,j,k]*Sx[i,j+1,k]
        PairCorryb_ijk = Sy[i,j,k]*Sy[i,j+1,k]
        PairCorrzb_ijk = Sz[i,j,k]*Sz[i,j+1,k]
    else:
        PairCorrxb_ijk = Sx[i,j,k]*Sx[i,0,k]
        PairCorryb_ijk = Sy[i,j,k]*Sy[i,0,k]
        PairCorrzb_ijk = Sz[i,j,k]*Sz[i,0,k]
        
    if k < L-1:
        PairCorrxc_ijk = Sx[i,j,k]*Sx[i,j,k+1]
        PairCorryc_ijk = Sy[i,j,k]*Sy[i,j,k+1]
        PairCorrzc_ijk = Sz[i,j,k]*Sz[i,j,k+1]
    else:
        PairCorrxc_ijk = Sx[i,j,k]*Sx[i,j,0]
        PairCorryc_ijk = Sy[i,j,k]*Sy[i,j,0]
        PairCorrzc_ijk = Sz[i,j,k]*Sz[i,j,0]

    return PairCorrxa_ijk, PairCorrya_ijk, PairCorrza_ijk, PairCorrxb_ijk, PairCorryb_ijk, PairCorrzb_ijk, PairCorrxc_ijk, PairCorryc_ijk, PairCorrzc_ijk

def Ecalc(x,i,j,k):
    theta, phi = x[0], x[1]
    Sx_ijk = S*np.sin(theta)*np.cos(phi)
    Sy_ijk = S*np.sin(theta)*np.sin(phi)
    Sz_ijk = S*np.cos(theta)
    Energy_ijk = 0

    Energy_ijk += -D*Sx_ijk**2

    #if Type[i,j,k]== 1:
    #    Energy_ijk += DMn[0]*Sx_ijk**2 + DMn[1]*Sy_ijk**2 + DMn[2]*Sz_ijk**2
    #else:
    #    Energy_ijk += DFe[0]*Sx_ijk**2 + DFe[1]*Sy_ijk**2 + DFe[2]*Sz_ijk**2
    #    #Energy_ijk += DFe_cubic*(Sx_ijk**2*Sy_ijk**2+Sy_ijk**2*Sz_ijk**2+Sz_ijk**2*Sx_ijk**2)
    if i < L-1:
        Energy_ijk += J*(Sx_ijk*Sx[i+1,j,k] + Sy_ijk*Sy[i+1,j,k] + Sz_ijk*Sz[i+1,j,k])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[0,j,k] + Sy_ijk*Sy[0,j,k] + Sz_ijk*Sz[0,j,k])
    if i > 0:
        Energy_ijk += J*(Sx_ijk*Sx[i-1,j,k] + Sy_ijk*Sy[i-1,j,k] + Sz_ijk*Sz[i-1,j,k])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[L-1,j,k] + Sy_ijk*Sy[L-1,j,k] + Sz_ijk*Sz[L-1,j,k])
        
    if j < L-1:
        Energy_ijk += J*(Sx_ijk*Sx[i,j+1,k] + Sy_ijk*Sy[i,j+1,k] + Sz_ijk*Sz[i,j+1,k])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[i,0,k] + Sy_ijk*Sy[i,0,k] + Sz_ijk*Sz[i,0,k])
    if j > 0:
        Energy_ijk += J*(Sx_ijk*Sx[i,j-1,k] + Sy_ijk*Sy[i,j-1,k] + Sz_ijk*Sz[i,j-1,k])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[i,L-1,k] + Sy_ijk*Sy[i,L-1,k] + Sz_ijk*Sz[i,L-1,k])
        
    if k < L-1:
        Energy_ijk += J*(Sx_ijk*Sx[i,j,k+1] + Sy_ijk*Sy[i,j,k+1] + Sz_ijk*Sz[i,j,k+1])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[i,j,0] + Sy_ijk*Sy[i,j,0] + Sz_ijk*Sz[i,j,0])
    if k > 0:
        Energy_ijk += J*(Sx_ijk*Sx[i,j,k-1] + Sy_ijk*Sy[i,j,k-1] + Sz_ijk*Sz[i,j,k-1])
    else:
        Energy_ijk += J*(Sx_ijk*Sx[i,j,L-1] + Sy_ijk*Sy[i,j,L-1] + Sz_ijk*Sz[i,j,L-1])
    return Energy_ijk

def Ecalc_theta_only(x,phi,i,j,k):
    this_e = Ecalc((x,phi),i,j,k)
    return this_e

def localField(x,i,j,k):
    theta, phi = x[0], x[1]
    Sx_ijk = S*np.sin(theta)*np.cos(phi)
    Sy_ijk = S*np.sin(theta)*np.sin(phi)
    Sz_ijk = S*np.cos(theta)
    localField_ijk = np.array([0,0,0])
    
    if i < L-1:
        localField_ijk = localField_ijk + J*np.array([Sx[i+1,j,k] , Sy[i+1,j,k] , Sz[i+1,j,k]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[0,j,k] , Sy[0,j,k] , Sz[0,j,k]])
    if i > 0:
        localField_ijk = localField_ijk + J*np.array([Sx[i-1,j,k] , Sy[i-1,j,k] , Sz[i-1,j,k]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[L-1,j,k] , Sy[L-1,j,k] , Sz[L-1,j,k]])
        
    if j < L-1:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j+1,k] , Sy[i,j+1,k] , Sz[i,j+1,k]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[i,0,k] , Sy[i,0,k] , Sz[i,0,k]])
    if j > 0:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j-1,k] , Sy[i,j-1,k] , Sz[i,j-1,k]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[i,L-1,k] , Sy[i,L-1,k] , Sz[i,L-1,k]])
        
    if k < L-1:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j,k+1] , Sy[i,j,k+1] , Sz[i,j,k+1]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j,0] , Sy[i,j,0] , Sz[i,j,0]])
    if k > 0:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j,k-1] , Sy[i,j,k-1] , Sz[i,j,k-1]])
    else:
        localField_ijk = localField_ijk + J*np.array([Sx[i,j,L-1] , Sy[i,j,L-1] , Sz[i,j,L-1]])
    return localField_ijk

def NonMinEcalc(x, NonMinE, i, j, k):
    return (Ecalc(x,i,j,k)-NonMinE)**2

#debye model for phonons
def phonon_probability(E,T):
    return phonon_prefactor(E,T)*unscaled_phonon_probability(E,T)

def unscaled_phonon_probability(E,T):
    if E < t_debye:
        #return E**2/(np.exp(E/T)-1)
        return E**8/(np.exp(E/T)-1)
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



#initialize the arrays
Sx = np.zeros((L,L,L))
Sy = np.zeros((L,L,L))
Sz = np.zeros((L,L,L))

Phi = np.zeros((L,L,L))
Theta = np.zeros((L,L,L))

Energy = np.zeros((L,L,L))

PairCorrxa = np.zeros((L,L,L))
PairCorrya = np.zeros((L,L,L))
PairCorrza = np.zeros((L,L,L))
PairCorrxb = np.zeros((L,L,L))
PairCorryb = np.zeros((L,L,L))
PairCorrzb = np.zeros((L,L,L))
PairCorrxc = np.zeros((L,L,L))
PairCorryc = np.zeros((L,L,L))
PairCorrzc = np.zeros((L,L,L))

#initialize the spin momentum vectors to have a random direction
for i in range(0,L):
    for j in range(0,L):
        for k in range(0,L):
            Phi[i,j,k] = np.random.uniform(0, 2*np.pi)
            Theta[i,j,k] = np.random.uniform(0, np.pi)
            Sx[i,j,k] = S*np.sin(Theta[i,j,k])*np.cos(Phi[i,j,k])
            Sy[i,j,k] = S*np.sin(Theta[i,j,k])*np.sin(Phi[i,j,k])
            Sz[i,j,k] = S*np.cos(Theta[i,j,k])
            Energy[i,j,k] = Ecalc((Theta[i,j,k],Phi[i,j,k]),i,j,k)

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

t_list = np.linspace(10,0.1,20)
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

    Energy_list = []
    Energy_list.append(np.sum(Energy))
    #now do the energy minimization procedure
    for shmoo in range(0,10):
        for i in range(0,L):
            for j in range(0,L):
                for k in range(0,L):
                    phonon_energy = inv_cdf(np.random.rand())
                    #print(localField((Theta[i,j,k],Phi[i,j,k]),i,j,k))
                    newtheta, newphi = spop.optimize.fmin(Ecalc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(Theta[i,j,k],Phi[i,j,k]), args = (i,j,k), disp=0)
                    
                    
                    print(t, i,j,k,newtheta, newphi, Ecalc((newtheta, newphi),i,j,k), phonon_energy)
                    Phi[i,j,k] = newphi
                    Theta[i,j,k] = newtheta
                    Sx[i,j,k] = S*np.sin(Theta[i,j,k])*np.cos(Phi[i,j,k])
                    Sy[i,j,k] = S*np.sin(Theta[i,j,k])*np.sin(Phi[i,j,k])
                    Sz[i,j,k] = S*np.cos(Theta[i,j,k])
                    Energy[i,j,k] = Ecalc((Theta[i,j,k],Phi[i,j,k]),i,j,k)


                    NonMinE = Energy[i,j,k]+phonon_energy
                    newtheta, newphi = spop.optimize.fmin(NonMinEcalc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(Theta[i,j,k],Phi[i,j,k]), args = (NonMinE,i,j,k), disp=0)
                    Phi[i,j,k] = newphi
                    Theta[i,j,k] = newtheta
                    Sx[i,j,k] = S*np.sin(Theta[i,j,k])*np.cos(Phi[i,j,k])
                    Sy[i,j,k] = S*np.sin(Theta[i,j,k])*np.sin(Phi[i,j,k])
                    Sz[i,j,k] = S*np.cos(Theta[i,j,k])
                    Energy[i,j,k] = Ecalc((Theta[i,j,k],Phi[i,j,k]),i,j,k)                
                    #newtheta, newphi = spop.optimize.fmin(NonMinEcalc, maxfun=5000, maxiter=5000, ftol=1e-6, xtol=1e-5, x0=(Theta[i,j,k],Phi[i,j,k]), args = (phonon_energy,i,j,k), disp=0)
                    
        Energy_list.append(np.sum(Energy))

    Emin_list.append(np.min(Energy_list))
    #print(Energy_list)
    #print(-L**3*2*J*S*S*3, -L**3*2*J*S*S*3-L**3*2*D*S*S)
    #print(phonon_energy)
    if 0: #should each 3d map of the spins be drawn?
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #plot solution
        for i in range(0,L):
            for j in range(0,L):
                for k in range(0,L):
                    ax.scatter(i, j, k, color = 'black', marker='o')
                    ax.plot([i,i+Sx[i,j,k]*moment_visualization_scale_factor], [j,j+Sy[i,j,k]*moment_visualization_scale_factor], [k,k+Sz[i,j,k]*moment_visualization_scale_factor], color = 'black')

    #calculate the pair correlations
    for i in range(0,L):
        for j in range(0,L):
            for k in range(0,L):
                PairCorrxa[i,j,k], PairCorrya[i,j,k], PairCorrza[i,j,k], PairCorrxb[i,j,k], PairCorryb[i,j,k], PairCorrzb[i,j,k], PairCorrxc[i,j,k], PairCorryc[i,j,k], PairCorrzc[i,j,k] = PairCorr(i,j,k)


    print("np.sum(PairCorrxa), np.sum(PairCorrya), np.sum(PairCorrza)")
    print(np.sum(PairCorrxa), np.sum(PairCorrya), np.sum(PairCorrza))

    print("np.sum(PairCorrxb), np.sum(PairCorryb), np.sum(PairCorrzb)")
    print(np.sum(PairCorrxb), np.sum(PairCorryb), np.sum(PairCorrzb))

    print("np.sum(PairCorrxc), np.sum(PairCorryc), np.sum(PairCorrzc)")
    print(np.sum(PairCorrxc), np.sum(PairCorryc), np.sum(PairCorrzc))

    pcxa_list.append(np.sum(PairCorrxa))
    pcya_list.append(np.sum(PairCorrya))
    pcza_list.append(np.sum(PairCorrza))

    pcxb_list.append(np.sum(PairCorrxb))
    pcyb_list.append(np.sum(PairCorryb))
    pczb_list.append(np.sum(PairCorrzb))

    pcxc_list.append(np.sum(PairCorrxc))
    pcyc_list.append(np.sum(PairCorryc))
    pczc_list.append(np.sum(PairCorrzc))

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
