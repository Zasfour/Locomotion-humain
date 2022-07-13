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

n = 500
taux = 5/n
T= linspace(0,5,n)


def d_xy (Xmoy, Ymoy, Xengendre, Yengendre):
    mn = 0
    for i in range (n):
        mn = mn + sqrt((Xmoy[i]-Xengendre[i])**2+(Ymoy[i]-Yengendre[i])**2)
    return mn

def d_theta (Theta_moy, Theta_engendre):
    mn = 0
    for i in range (n):
        mn = mn + abs(Theta_moy[i]-Theta_engendre[i])
    return mn


options = {'maxfev': 100000 , 'rhobeg' : 0.01 , 'rhoend' : 1e-6}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0, 0,0,0,0,0], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0,0, 0,0,0,0,0], 0, 1))  
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0], 0, 1))    



def MH_DOC (c1,c2,c3,c4, Xi,Xf):           
    xi = Xi[0]
    yi = Xi[1]
    thetai = Xi[2]
    
    xf = Xf[0]
    yf = Xf[1]
    thetaf = Xf[2]
    
    opti = casadi.Opti()  

    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)
 
    v1 = opti.variable(n)       
    v2 = opti.variable(n)        
    w = opti.variable(n)         

    u1 = opti.variable(n)      
    u3 = opti.variable(n)       
    u2 = opti.variable(n)      


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *u3 ,u3 ) + dot(c4 *(theta-thetaf) ,theta-thetaf ) ) )   

        
    opti.subject_to( x[0] == xi )       
    opti.subject_to( y[0] == yi )    
    opti.subject_to( theta[0] == thetai )


    opti.subject_to( v1[0] == 0.000 )
    opti.subject_to( w[0] == 0.000 )
    opti.subject_to( v2[0] == 0.00 )
    opti.subject_to( v1[-1] == 0.0 )
    opti.subject_to( w[-1] == 0.00 )
    opti.subject_to( v2[-1] == 0.0 )

    opti.subject_to( u1[-1] == 0.0 )
    opti.subject_to( u2[-1] == 0.0 )
    opti.subject_to( u3[-1] == 0.0 )

    opti.subject_to( u1[0] == 0.00 )
    opti.subject_to( u2[0] == 0.00 )
    opti.subject_to( u3[0] == 0.00 )

 
    opti.subject_to( x[1:]  == x[:n-1]+taux*(np.cos(theta[:n-1])*v1[:n-1] - np.sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(np.sin(theta[:n-1])*v1[:n-1] + np.cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]  == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )


    opti.subject_to( x[-1]==xf )
    opti.subject_to( y[-1]==yf )
    opti.subject_to( theta[-1]==thetaf )


    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001})

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    
    return X1_1,X2_1,X3_1




def MH_PDFO (C):
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5,D0,D1,D2,D3,D4,D5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    c3 = C0* (T**0) + C1* (T**1) + C2* (T**2) + C3* (T**3) + C4* (T**4) + C5* (T**5)
    c4 = D0* (T**0) + D1* (T**1) + D2* (T**2) + D3* (T**3) + D4* (T**4) + D5* (T**5)

    print(C)
    
    mk = 0
    m01 = 0
    
    for j in range (c1.shape[0]):
        if c1[j] < 0 :
            c1[j] = - c1[j] 
            mk = mk - float(c1[j]) 
        if c2[j] < 0 :
            c2[j] = - c2[j]
            mk = mk - float(c2[j]) 
        if c3[j] < 0 :
            c3[j] = - c3[j]
            mk = mk - float(c3[j]) 
        if c4[j] < 0 :
            c4[j] = - c4[j]
            mk = mk - float(c4[j])

    for j in range (40):
        T0 = np.loadtxt(data[j])
        Xmoy = T0[0]
        Ymoy = T0[1]
        Theta_moy = T0[5]
        Xi = [Xmoy[0],Ymoy[0],Theta_moy[0]]
        Xf = [Xmoy[-1],Ymoy[-1],Theta_moy[-1]]
        
        
        X1_1, X2_1, X3_1  = MH_DOC (c1,c2,c3,c4, Xi,Xf)
        
        m01 += float(d_xy (Xmoy, Ymoy, X1_1, X2_1) + d_theta (Theta_moy, X3_1))

    m01 += 400*float(abs(np.sum(c1 + c2 + c3 ) - n)) + 400* mk
       
    return m01



res = pdfo( MH_PDFO, [1/3, 0, 0, 0, 0, 0,1/3, 0,0,0,0,0,1/3, 0,0,0,0,0,0,0,0,0,0,0], constraints=Lin_const, options=options)


A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO,D0_PDFO,D1_PDFO,D2_PDFO,D3_PDFO,D4_PDFO,D5_PDFO = res.x
    
c1_PDFO = A0_PDFO* T**0+ A1_PDFO* T + A2_PDFO * T**2 + A3_PDFO* T**3 + A4_PDFO* T**4 + A5_PDFO* T**5
c2_PDFO = B0_PDFO* T**0+ B1_PDFO* T + B2_PDFO * T**2 + B3_PDFO* T**3 + B4_PDFO* T**4 + B5_PDFO* T**5
c3_PDFO = C0_PDFO* T**0+ C1_PDFO* T + C2_PDFO * T**2 + C3_PDFO* T**3 + C4_PDFO* T**4 + C5_PDFO* T**5
c4_PDFO = D0_PDFO* T**0+ D1_PDFO* T + D2_PDFO * T**2 + D3_PDFO* T**3 + D4_PDFO* T**4 + D5_PDFO* T**5



RMSE_plan = np.zeros(40)
RMSE_ang = np.zeros(40)


for i in range (40):   
    T0 = np.loadtxt(data[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = T0[5]
    Xi = [Xmoy[0],Ymoy[0],Theta_moy[0]]
    Xf = [Xmoy[-1],Ymoy[-1],Theta_moy[-1]]

    X,Y,Theta = MH_DOC (c1_PDFO, c2_PDFO,c3_PDFO,c4_PDFO, Xi,Xf)
    
    RMSE_plan[i] = np.sqrt((np.linalg.norm(Xmoy-X)**2 + np.linalg.norm(Ymoy-Y)**2 )/n)
    RMSE_ang[i] = np.sqrt(( np.linalg.norm(Theta_moy-Theta)**2 )/n)
    
    
    plt.figure (figsize = (20,15))
    plt.plot(Xmoy, Ymoy ,color = 'r', label = 'Trajectoire moyenne de {}'.format(data[i]))
    plt.plot(X, Y, color= 'green', label = 'PDFO' )
    plt.xlabel ('X [m]')
    plt.ylabel ('Y [m]')
    plt.legend()
    plt.savefig("POIDS_VARIE_Theta_Traj_moy_{}.png".format(data[i]))
    
    
df = pd.DataFrame({'Mean_traj (Bi-level PDFO)' : data, 'RMSE_plan_unity [m]' : RMSE_plan,
                   'RMSE_angular_unity [rad]' : RMSE_ang, 'RMSE_angular_unity [degree]' : RMSE_ang*(180/np.pi)})


df.to_csv('RMSE_poids_variable_Theta_Trajectoire_moyenne.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_variable_Theta_Trajectoire_moyenne.csv'), 'MH_poids_varie_Theta_Trajectoire_moyenne.png')
