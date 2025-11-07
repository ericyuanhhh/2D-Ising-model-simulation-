# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 08:53:52 2021

@author: EricYuan
"""



import random
from pylab import plot,figure,errorbar,hist,subplot,title,xlabel,ylabel,legend
import matplotlib.pyplot as plt
from numpy import mod,zeros,mean,arange,sign, exp,save,array
## ising model
import time
from numba import jit

J = 1
L_x = L_y = 10 #dimension
low_T = 0.015*J      #lower temperature limit
high_T = 4.5*J    #higher temp limit
delta_T=0.015*J   

latt = zeros([L_x,L_y])
# initialize the spin configuration

for num_x in range(L_y):
    for num_y in range(L_y):
        latt[num_x,num_y] = random.choice([-1,1])
        latt[num_x,num_y] = sign(random.random()-0.5)
# calculate the energy difference 

@jit
def get_delta_E(nx,ny):
    nx_ml= mod(nx-1,L_x)   # to fulfill the periodic boundary condition
    nx_pl = mod(nx+1,L_x)  
    ny_ml = mod(ny-1,L_y)
    ny_pl = mod(ny+1,L_y)
    delta_E = 2*J*latt[nx,ny]*(latt[nx_ml,ny]+latt[nx_pl,ny]+latt[nx,ny_ml]+latt[nx,ny_pl])
    
    return delta_E

# Update the spin configuration using local update

def one_sweep(T):
    for num_x in range(L_x):
        for num_y in range(L_y):
            if exp(-get_delta_E(num_x,num_y)/T) > random.random():
                latt[num_x,num_y] = -latt[num_x,num_y]
#calculate the energy of a configuration

def one_random_sweep(T):
    num_x = random.randint(0,L_x -1)
    num_y = random.randint(0,L_y-1)
    if exp(-get_delta_E(num_x,num_y)/T) > random.random():
               latt[num_x,num_y] = -latt[num_x,num_y]    

def one_measure_sweep(T):
    for j in range(10):
        num_x = random.randint(0,L_x -1)
        num_y = random.randint(0,L_y-1)
        if exp(-get_delta_E(num_x,num_y)/T) > random.random():
                   latt[num_x,num_y] = -latt[num_x,num_y]               

@jit            
def cal_energy():
    E = 0
    for nx in range(L_x):
        for ny in range(L_y):
            nx_ml= mod(nx-1,L_x)
            nx_pl = mod(nx+1,L_x)
            ny_ml = mod(ny-1,L_y)
            ny_pl = mod(ny+1,L_y)
            E_x_y = -J*latt[nx,ny]*(latt[nx_ml,ny]+latt[nx_pl,ny]+latt[nx,ny_ml]+latt[nx,ny_pl])
            E +=E_x_y
    E = E/2
    E_2 = E**2   # square of energy
    return E,E_2
@jit
def main_mode():
    start = time.time()
    e_sweep = 10**5
    m_sweep = 3*10**4
    T_list = arange(low_T,high_T,delta_T)
    M_list = zeros([len(T_list),m_sweep])
    M_abs_list = zeros([len(T_list),m_sweep])
    mean_M_abs =  zeros([len(T_list)])
    mean_M = zeros([len(T_list)])
    num_T = 0
    E_list = zeros([len(T_list),m_sweep])
    mean_E = zeros([len(T_list)])
    E_sqrt_list = zeros([len(T_list),m_sweep]) 
    mean_E_sqrt = zeros([len(T_list)]) 
    M_sqrt_list = zeros([len(T_list),m_sweep]) 
    mean_M_sqrt = zeros([len(T_list)]) 
    M_abs_sqrt_list = zeros([len(T_list),m_sweep]) 
    mean_M_abs_sqrt = zeros([len(T_list)])
    for nT in T_list:
        for nb in range(e_sweep):
            one_random_sweep(nT)      # update the spin configuration until it reach equilibirum
        print("start measurement at "+ str(int(nT*1000)))
        for i in range(m_sweep):
            one_measure_sweep(nT)
            M = sum(sum(latt))/L_x/L_y   # calculate the magnetization
            M_list[num_T,i] = M
            M_abs_list[num_T,i] = abs(M)
            E_list[num_T,i],E_sqrt_list[num_T,i] = cal_energy() # find energy
            M_sqrt_list[num_T,i] = M**2   # square of magnetization 
            M_abs_sqrt_list[num_T,i] = (abs(M))**2
        #calculate the mean of each observables 
        mean_M[num_T] = mean(M_list[num_T,:])
        mean_M_abs[num_T] = mean(M_abs_list[num_T,:])
        mean_M_sqrt[num_T] = mean(M_sqrt_list[num_T,:])
        mean_E[num_T] = mean(E_list[num_T,:])
        mean_M_abs_sqrt[num_T] = mean(M_abs_sqrt_list[num_T,:])
        mean_E_sqrt[num_T] = mean(E_sqrt_list[num_T,:])
        num_T +=1
    save("mean_E_10_random.npy",mean_E)
    save("mean_E_sqrt_10_random.npy", mean_E_sqrt)
    c = (mean_E_sqrt-mean_E**2)/(L_x*L_y*T_list**2) #heat capacity]
    save("heatcap_10_random.npy",c)
    sus = (mean_M_abs_sqrt-mean_M_abs**2)*(L_x*L_y)/T_list
    save("mean_M_sqrt_10_random.npy",mean_M_sqrt)
    save("mean_M_10_random.npy",mean_M)
    save("mean_M_abs_sqrt_10_random.npy",mean_M_abs_sqrt)
    save("mean_M_abs_10_random.npy",mean_M_abs)
    save("sus_10_random.npy",sus)
    
    
    
    # plot the result.
    end = time.time()
    print(end-start)
    
    fig, ax = plt.subplots(2, 2,constrained_layout=True)
    fig.suptitle('Dimension = 10x10')
    
    ax[0, 0].plot(T_list,mean_M_abs,'ro',label = str(L_x)+"x"+str(L_y),markersize=3) #row=0, col=0
    ax[0,0].set_title('Magnetization vs temperature',fontsize = 9)
    ax[0,0].set_xlabel('Temperature')
    ax[0,0].set_ylabel('Magnetization')
    
    ax[1, 0].plot(T_list,mean_E,'bo',label = str(L_x)+"x"+str(L_y),markersize=3) #row=1, col=0
    ax[1,0].set_title('Average total energy vs temperature',fontsize = 9)
    ax[1,0].set_xlabel("Temperature",fontsize=9)
    ax[1,0].set_ylabel("Average total nergy",fontsize=9)
    
    
    ax[0, 1].plot(T_list,sus,'go',label = str(L_x)+"x"+str(L_y),markersize=3) #row=0, col=1
    ax[0,1].set_title('Susceptibility vs temperature',fontsize = 9)
    ax[0,1].set_xlabel('Temperature',fontsize=9)
    ax[0,1].set_ylabel("Susceptibility",fontsize=9)
    
    
    
    ax[1, 1].plot(T_list,c,'co',label = str(L_x)+"x"+str(L_y),markersize=3)#row=1, col=1
    ax[1,1].set_title('Heat capacity vs temperature',fontsize = 9)
    ax[1,1].set_xlabel("Temperature",fontsize=9)
    ax[1,1].set_ylabel("specific heat",fontsize=9)
    plt.savefig('16x16_ising.png',dpi = 300)

main_mode()
    
    
    