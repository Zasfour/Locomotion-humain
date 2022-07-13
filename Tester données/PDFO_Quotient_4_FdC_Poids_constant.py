#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *
import dataframe_image as dfi




n = 500
taux = 5/n
T = linspace(0,5,n)

data = ['E-0615.dat','E-0640.dat','E0615.dat','E0640.dat','E1500.dat','E1515.dat','E1540.dat','E4000.dat','E4015.dat','E4040.dat',
        'N-0615.dat','N-0640.dat','N0615.dat','N0640.dat','N1500.dat','N1515.dat','N1540.dat','N4000.dat','N4015.dat','N4040.dat',
        'O-0615.dat','O-0640.dat','O0615.dat','O0640.dat','O1500.dat','O1515.dat','O1540.dat','O4000.dat','O4015.dat','O4040.dat',
        'S-0615.dat','S-0640.dat','S0615.dat','S0640.dat','S1500.dat','S1515.dat','S1540.dat','S4000.dat','S4015.dat','S4040.dat']


def orientation (x,y,theta, r, i):
    X00 = [x,x+r*cos(theta)]
    Y00 = [y,y+r*sin(theta)]
    
    
    if i == 1 :
        plt.plot(X00,Y00,'r',label='Orientation')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'r' )
        plt.legend()
    else :
        plt.plot(X00,Y00,'r')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'r' )
        

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r



x = SX.sym ('x', n)
y = SX.sym ('y', n)
xf = SX.sym('xf',1)
yf = SX.sym('yf',1)

M000 = SX.zeros(1)
Y = ((xf - x[1:])**2 + (yf - y[1:])**2 + 10**(-5) )/((xf - x[:-1])**2 + (yf - y[:-1])**2 + 10**(-5) )
for i in range (Y.shape[0]):
    M000 += Y[i]
                                                
Direct = Function('Direct', [x,xf,y,yf],[M000])



def MH_DOC(c1,c2,c3,c4,Xi,Xf):
    
    opti = casadi.Opti()   
    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    w = opti.variable(n)         ## vitesse angulaire
    
        ## les vitesses 
    u1 = opti.variable(n)        ## accélération latérale
    u3 = opti.variable(n)        ## accélération orthogonal
    u2 = opti.variable(n)        ## accélération angulaire
    
    
    opti.minimize(  taux*( c1 *dot(u1,u1) + c2*dot(u2,u2 ) + c3*dot(u3 ,u3 ) +  c4 * Direct(y,Xf[1],x,Xf[0]) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0.000 )
    opti.subject_to( w[0]  == 0.000 )
    opti.subject_to( v2[0] == 0.000 )
    opti.subject_to( u1[0] == 0.000 )
    opti.subject_to( u2[0] == 0.000 )
    opti.subject_to( u3[0] == 0.000 )
    

    ## pour les contraintes d'égaliter
    opti.subject_to( x[1:]  == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]  == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )

    ## pour les conditions finales
    opti.subject_to( x[-1]== Xf[0] )
    opti.subject_to( y[-1]== Xf[1] )
    opti.subject_to( theta[-1]== Xf[2] )
    opti.subject_to( v1[-1] == 0.000 ) 
    opti.subject_to( w[-1]  == 0.000 ) 
    opti.subject_to( v2[-1] == 0.000 )
    opti.subject_to( u1[-1] == 0.000 )
    opti.subject_to( u2[-1] == 0.000 )
    opti.subject_to( u3[-1] == 0.000 )
    
    opti.callback(lambda i: plt.plot(opti.debug.value(x),opti.debug.value(y), label = "{}".format(i)))
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta)


options = {'maxfev': 10000  , 'rhoend' : 1e-6}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1] , [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1,1], 1, 1)

def MH_PDFO (C):
    c1,c2,c3,c4 = C
    print(C)

    mk = 0

    if c1 < 0 :
        c1 = -c1
        mk += c1
    if c2 < 0 :
        c2 = -c2
        mk += c2
    if c3 < 0 :
        c3 = -c3
        mk += c3
    if c4 < 0 :
        c4 = -c4
        mk += c4
    
    opti = casadi.Opti()   
    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    w = opti.variable(n)         ## vitesse angulaire
    
        ## les vitesses 
    u1 = opti.variable(n)        ## accélération latérale
    u3 = opti.variable(n)        ## accélération orthogonal
    u2 = opti.variable(n)        ## accélération angulaire
    
    
    opti.minimize(  taux*( c1 *dot(u1,u1) + c2*dot(u2,u2 ) + c3*dot(u3 ,u3 ) +  c4 * Direct(y,Xmoy[-1],x,Ymoy[-1]) ) )     
    
    opti.subject_to( x[0] == Xmoy[0] )        
    opti.subject_to( y[0] == Ymoy[0] )
    opti.subject_to( theta[0] == Theta_moy[0] )
    opti.subject_to( v1[0] == 0.000 )
    opti.subject_to( w[0]  == 0.000 )
    opti.subject_to( v2[0] == 0.000 )
    opti.subject_to( u1[0] == 0.000 )
    opti.subject_to( u2[0] == 0.000 )
    opti.subject_to( u3[0] == 0.000 )
    

    ## pour les contraintes d'égaliter
    opti.subject_to( x[1:]  == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]  == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )

    ## pour les conditions finales
    opti.subject_to( x[-1]== Xmoy[-1] )
    opti.subject_to( y[-1]== Ymoy[-1] )
    opti.subject_to( theta[-1]== Theta_moy[-1] )
    opti.subject_to( v1[-1] == 0.000 ) 
    opti.subject_to( w[-1]  == 0.000 ) 
    opti.subject_to( v2[-1] == 0.000 )
    opti.subject_to( u1[-1] == 0.000 )
    opti.subject_to( u2[-1] == 0.000 )
    opti.subject_to( u3[-1] == 0.000 )
    
    opti.callback(lambda i: plt.plot(opti.debug.value(x),opti.debug.value(y), label = "{}".format(i)))
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    m01 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    m02 = 10* abs (c1 + c2 + c3 + c4 -1)
    m03 = 10* mk
    
    m1 = float(m01+m02+m03)

    return m1

RMSE_plan = np.zeros(40)
RMSE_ang = np.zeros(40)

for i in range (40):

    T0 = np.loadtxt(data[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = T0[5]

    res = pdfo( MH_PDFO, [1/5,2/5,1/5,1/5],bounds=bounds1, constraints=lin_con1, options=options)

    c1,c2,c3,c4 = res.x

    Xi = [Xmoy[0],Ymoy[0],Theta_moy[0]]
    Xf = [Xmoy[-1],Ymoy[-1],Theta_moy[-1]]

    x,y,o = MH_DOC(c1,c2,c3,c4,Xi,Xf)

    plt.figure(figsize=(20,15))
    plt.plot(Xmoy,Ymoy,'r',label = 'Traj moyenne {}'.format(data[i]))
    plt.plot(x,y, label = 'PDFO')
    plt.plot(x[0],y[0],'o',label = "Point initial")
    plt.plot(x[-1],y[-1],'o',label = "Point final")
    orientation (x[0],y[0],o[0],  0.005, 0)
    orientation (x[100],y[100],o[100],  0.005, 0)
    orientation (x[200],y[200],o[200],  0.005, 0)
    orientation (x[300],y[300],o[300],  0.005, 0)
    orientation (x[400],y[400],o[400], 0.005, 0)
    orientation (x[-1],y[-1],o[-1],  0.005, 1)
    plt.legend()
    plt.xlabel("X[m]")
    plt.ylabel("Y[m]")
    plt.grid()
    plt.savefig('{}.png'.format(data[i]))

    RMSE_plan[i], RMSE_ang[i] = sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 )/n), sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)


df = pd.DataFrame({'Mean_traj (Bi-level PDFO)' : data, 'RMSE_plan_unity [m]' : RMSE_plan,
                   'RMSE_angular_unity [rad]' : RMSE_ang, 'RMSE_angular_unity [degree]' : RMSE_ang*(180/np.pi)})


df.to_csv('RMSE_poids_constant_4FdC.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_constant_4FdC.csv'), 'MH_poids_constant_4FdC.png')