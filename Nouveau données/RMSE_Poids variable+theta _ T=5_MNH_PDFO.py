#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from casadi import *
from pdfo import *
import dataframe_image as dfi



data = ['E-0615.dat','E-0640.dat','E0615.dat','E0640.dat','E1500.dat','E1515.dat','E1540.dat','E4000.dat','E4015.dat','E4040.dat',
        'N-0615.dat','N-0640.dat','N0615.dat','N0640.dat','N1500.dat','N1515.dat','N1540.dat','N4000.dat','N4015.dat','N4040.dat',
        'O-0615.dat','O-0640.dat','O0615.dat','O0640.dat','O1500.dat','O1515.dat','O1540.dat','O4000.dat','O4015.dat','O4040.dat',
        'S-0615.dat','S-0640.dat','S0615.dat','S0640.dat','S1500.dat','S1515.dat','S1540.dat','S4000.dat','S4015.dat','S4040.dat']



i2 = [1,3,6,7,8,9,11,13,16,18,19,21,23,26,27,28,29,31,33,36,37,38,39]   ##### 23

data2 = []

for i in range(40):
    if i in i2 :
        data2.append(data[i])



n = 500
taux = 5/n
T= linspace(0,5,n)




def MNH_DOC(c1,c2,c3):

    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2)+dot(c3*(x3-Theta_moy[-1]),x3-Theta_moy[-1]))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==Xmoy[0] )        
    opti.subject_to( x2[0]==Ymoy[0] )
    opti.subject_to( x3[0]==Theta_moy[0] )

    opti.subject_to( u1[0] == 0 )
    opti.subject_to( u2[0] == 0 )

    opti.subject_to( u1[-1] == 0)
    opti.subject_to( u2[-1] == 0)


    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:]  - x1[:n-1])/taux )
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:]  - x2[:n-1])/taux )
    opti.subject_to( u2[:n-1] ==(x3[1:]  - x3[:n-1])/taux)

    ## pour les conditions finales
    opti.subject_to( x1[-1]==Xmoy[-1] )
    opti.subject_to( x2[-1]==Ymoy[-1] )
    opti.subject_to( x3[-1]==Theta_moy[-1] )

    #p_opts = dict(print_time = False, verbose = False)
    #s_opts = dict(print_level = 0)


    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001}  ) #, p_opts,s_opts)      


    sol = opti.solve()
    
    return sol.value(x1), sol.value(x2),sol.value(x3)



######################################## Bi-level PDFO

options = {'maxfev': 1000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0], 0, 1))  
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0,0,0,0,0,0,0], 0, 1))        




def Unicycle (C) :
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    c3 = C0* (T**0) + C1* (T**1) + C2* (T**2) + C3* (T**3) + C4* (T**4) + C5* (T**5)
    
    print(C)

    
    mk = 0
    
    for j in range (c1.shape[0]):
        if c1[j] < 0 :
            c1[j] = - c1[j] 
            mk = mk - c1[j] 
        if c2[j] < 0 :
            c2[j] = - c2[j]
            mk = mk - c2[j] 
        if c3[j] < 0 :
            c3[j] = - c3[j]
            mk = mk - c3[j]             
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2)+dot(c3*(x3-Theta_moy[-1]),x3-Theta_moy[-1]))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==Xmoy[0])        
    opti.subject_to( x2[0]==Ymoy[0] )
    opti.subject_to( x3[0]==Theta_moy[0] )
    
    opti.subject_to( u1[0] == 0 )
    opti.subject_to( u2[0] == 0 )
    opti.subject_to( u1[-1] == 0)
    opti.subject_to( u2[-1] == 0)
    

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:]  - x1[:n-1])/taux)
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:]  - x2[:n-1])/taux)
    opti.subject_to( u2[:n-1] ==(x3[1:]  - x3[:n-1])/taux)
    
    ## pour les conditions finales
    opti.subject_to( x1[-1]==Xmoy[-1] )
    opti.subject_to( x2[-1]==Ymoy[-1] )
    opti.subject_to( x3[-1]==Theta_moy[-1] )
    
    opti.solver('ipopt', {"expand":True},{'acceptable_constr_viol_tol':0.0001})
    
    sol = opti.solve() 
    
    X1_1 = opti.debug.value(x1)
    X2_1 = opti.debug.value(x2)
    X3_1 = opti.debug.value(x3)
    
    m01 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    
    m02 = 10*np.abs(np.sum(c1 + c2 + c3) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return float(m1)



RMSE_plan = np.zeros(23)
RMSE_ang = np.zeros(23)


for i in range (23):
    T0 = np.loadtxt(data2[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = atan(T0[3]/T0[2])

    M00 = vertcat(0,Theta_moy[1:])

    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    U1_moy = cos(Theta_moy)*T0[2]+T0[3]*sin(Theta_moy)

    U2_moy = (M00-Theta_moy)/taux
    
    res = pdfo( Unicycle, [1/3, 0, 0, 0, 0, 0,1/3, 0,0,0,0,0,1/3, 0,0,0,0,0], constraints=Lin_const, options=options) 
    
    A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO = res.x
    
    c1_PDFO = A0_PDFO* T**0+ A1_PDFO* T + A2_PDFO * T**2 + A3_PDFO* T**3 + A4_PDFO* T**4 + A5_PDFO* T**5
    c2_PDFO = B0_PDFO* T**0+ B1_PDFO* T + B2_PDFO * T**2 + B3_PDFO* T**3 + B4_PDFO* T**4 + B5_PDFO* T**5
    c3_PDFO = C0_PDFO* T**0+ C1_PDFO* T + C2_PDFO * T**2 + C3_PDFO* T**3 + C4_PDFO* T**4 + C5_PDFO* T**5
    
    X,Y,THETA  = MNH_DOC (c1_PDFO,c2_PDFO,c3_PDFO)
    
    RMSE_plan[i], RMSE_ang[i]  = sqrt((np.linalg.norm(Xmoy-X)**2 + np.linalg.norm(Ymoy-Y)**2 )/n), sqrt( np.linalg.norm(Theta_moy-THETA)**2 /n)

    
df = pd.DataFrame({'Mean_traj (Non_holonomic model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : RMSE_plan,
                   'RMSE_angular_unity [rad]' : RMSE_ang, 'RMSE_angular_unity [degree]' : RMSE_ang* (180/np.pi)})

df.to_csv('RMSE_poids_variable_theta_PDFO_MNH_T=5.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_variable_theta_PDFO_MNH_T=5.csv'), 'RMSE_poids_variable_theta_PDFO_MNH_T=5.png')