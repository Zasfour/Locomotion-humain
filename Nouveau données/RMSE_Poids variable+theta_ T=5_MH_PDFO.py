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
T= np.linspace(0,5,n)




def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 

def MH_DOC (c1,c2,c3,c4, Xi,Xf):
    x0 = Xi[0]
    y0 = Xi[1]
    theta0 = Xi[2]
    
    xf = Xf[0]
    yf = Xf[1]
    thetaf = Xf[2]
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    w = opti.variable(n)         ## vitesse angulaire


    ## les accélération 
    u1 = opti.variable(n)        ## accélération latérale
    u3 = opti.variable(n)        ## accélération orthogonal
    u2 = opti.variable(n)        ## accélération angulaire


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *  u3 ,u3 ) + dot(c4 *  (theta-thetaf) ,theta-thetaf ) ) )    # ma fonction objetion

        # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == x0 + 10**(-4))       
    opti.subject_to( y[0] == y0 + 10**(-4))    
    opti.subject_to( theta[0] == theta0 + 10**(-4))


    opti.subject_to( v1[0] == 0.0001 )
    opti.subject_to( w[0] == 0.0001 )
    opti.subject_to( v2[0] == 0.0001 )
    opti.subject_to( v1[-1] == 0.0001 )
    opti.subject_to( w[-1] == 0.0001 )
    opti.subject_to( v2[-1] == 0.0001 )

    opti.subject_to( u1[-1] == 0.0001 )
    opti.subject_to( u2[-1] == 0.0001 )
    opti.subject_to( u3[-1] == 0.0001 )

    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u3[0] == 0.0001 )



        ## pour les contraintes d'égaliter
    opti.subject_to( x[1:] + 10**(-4) == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] + 10**(-4) == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:] + 10**(-4) == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] + 10**(-4))  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] + 10**(-4)) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] + 10**(-4)) )


        ## pour les conditions finales
    opti.subject_to( x[-1]==xf + 10**(-4))
    opti.subject_to( y[-1]==yf + 10**(-4))
    opti.subject_to( theta[-1]==thetaf + 10**(-4))


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    return sol.value(x),sol.value(y),sol.value(theta)
    



options = {'maxfev': 500 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0, 0,0,0,0,0], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0, 0,0,0,0,0,0, 0,0,0,0,0], 0, 1))    
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0], 0, 1))    


def MH_PDFO (C):
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5,D0,D1,D2,D3,D4,D5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    c3 = C0* (T**0) + C1* (T**1) + C2* (T**2) + C3* (T**3) + C4* (T**4) + C5* (T**5)
    c4 = D0* (T**0) + D1* (T**1) + D2* (T**2) + D3* (T**3) + D4* (T**4) + D5* (T**5)
     
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
        if c4[j] < 0 :
            c4[j] = - c4[j]
            mk = mk - c4[j]             
            
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    w = opti.variable(n)         ## vitesse angulaire


    ## les accélération 
    u1 = opti.variable(n)        ## accélération latérale
    u3 = opti.variable(n)        ## accélération orthogonal
    u2 = opti.variable(n)        ## accélération angulaire


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *u3 ,u3 ) + dot(c4 *(theta-Theta_moy[-1]) ,theta-Theta_moy[-1] ) ) )   

        # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == Xmoy[0] )       
    opti.subject_to( y[0] == Ymoy[0] )    
    opti.subject_to( theta[0] == Theta_moy[0] )


    opti.subject_to( v1[0] == 0 )
    opti.subject_to( w[0] ==  0 )
    opti.subject_to( v2[0] == 0 )
    opti.subject_to( v1[-1] == 0 )
    opti.subject_to( w[-1] == 0 )
    opti.subject_to( v2[-1] == 0 )

    opti.subject_to( u1[-1] == 0 )
    opti.subject_to( u2[-1] == 0 )
    opti.subject_to( u3[-1] == 0 )

    opti.subject_to( u1[0] == 0 )
    opti.subject_to( u2[0] == 0 )
    opti.subject_to( u3[0] == 0 )



        ## pour les contraintes d'égaliter
    opti.subject_to( x[1:]  == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]  == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )


        ## pour les conditions finales
    opti.subject_to( x[-1]==Xmoy[-1] )
    opti.subject_to( y[-1]==Ymoy[-1] )
    opti.subject_to( theta[-1]==Theta_moy[-1] )


    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001})

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    
    m01 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    
    m02 = 10*abs(np.sum(c1 + c2 + c3 + c4) - n)
    
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

    Xi = [Xmoy[0],Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1],Ymoy[-1], Theta_moy[-1]]

    
    res = pdfo( MH_PDFO, [1/3, 0, 0, 0, 0, 0,1/3, 0,0,0,0,0,1/3, 0,0,0,0,0,0, 0,0,0,0,0], constraints=Lin_const, options=options) 
    
    A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO,D0_PDFO,D1_PDFO,D2_PDFO,D3_PDFO,D4_PDFO,D5_PDFO = res.x
    
    c1_PDFO = A0_PDFO* T**0+ A1_PDFO* T + A2_PDFO * T**2 + A3_PDFO* T**3 + A4_PDFO* T**4 + A5_PDFO* T**5
    c2_PDFO = B0_PDFO* T**0+ B1_PDFO* T + B2_PDFO * T**2 + B3_PDFO* T**3 + B4_PDFO* T**4 + B5_PDFO* T**5
    c3_PDFO = C0_PDFO* T**0+ C1_PDFO* T + C2_PDFO * T**2 + C3_PDFO* T**3 + C4_PDFO* T**4 + C5_PDFO* T**5
    c4_PDFO = D0_PDFO* T**0+ D1_PDFO* T + D2_PDFO * T**2 + D3_PDFO* T**3 + D4_PDFO* T**4 + D5_PDFO* T**5

    
    X,Y,THETA  = MH_DOC (c1_PDFO,c2_PDFO,c3_PDFO,c4_PDFO, Xi,Xf)
    
    RMSE_plan[i], RMSE_ang[i]  = sqrt((np.linalg.norm(Xmoy-X)**2 + np.linalg.norm(Ymoy-Y)**2 )/n), sqrt( np.linalg.norm(Theta_moy-THETA)**2 /n)

   
df = pd.DataFrame({'Mean_traj (Holonomic model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : RMSE_plan,
                   'RMSE_angular_unity [rad]' : RMSE_ang, 'RMSE_angular_unity [degree]' : RMSE_ang* (180/np.pi)})

df.to_csv('RMSE_poids_variable_theta_PDFO_MH_T=5.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_variable_theta_PDFO_MH_T=5.csv'), 'RMSE_poids_variable_theta_PDFO_MH_T=5.png')
