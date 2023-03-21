#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *



def Orientation_ (x,y,theta, m, couleur) :
    plt. arrow(x, y, (m/2)*cos(theta),(m/2)*sin(theta), color = couleur, head_width = m/2, head_length = m)



def plot_traj(name,X,Y,O):

    mm0 = np.loadtxt(name)
    mn = ((mm0[0][0]-mm0[0][-1])**2 + (mm0[1][0]-mm0[1][-1])**2)**(1/2)/30
    x = mm0[0]
    y = mm0[1]
    o = mm0[5]
    Orientation_ (x[0],y[0],o[0] ,mn, 'olive')
    Orientation_ (x[50],y[50],o[50] ,mn, 'olive')
    Orientation_ (x[100],y[100],o[100] ,mn,'olive')
    Orientation_ (x[150],y[150],o[150] ,mn, 'olive')
    Orientation_ (x[200],y[200],o[200] ,mn, 'olive')
    Orientation_ (x[250],y[250],o[250] ,mn, 'olive')
    Orientation_ (x[300],y[300],o[300] ,mn, 'olive')
    Orientation_ (x[350],y[350],o[350] ,mn, 'olive')
    Orientation_ (x[400],y[400],o[400] ,mn, 'olive')
    Orientation_ (x[450],y[450],o[450] ,mn, 'olive')
    Orientation_ (x[-1],y[-1],o[-1], mn, 'olive')
    
    Orientation_ (X[0],Y[0],O[0] ,mn, 'orangered')
    Orientation_ (X[50],Y[50],O[50] ,mn, 'orangered')
    Orientation_ (X[100],Y[100],O[100] ,mn,'orangered')
    Orientation_ (X[150],Y[150],O[150] ,mn, 'orangered')
    Orientation_ (X[200],Y[200],O[200] ,mn, 'orangered')
    Orientation_ (X[250],Y[250],O[250] ,mn, 'orangered')
    Orientation_ (X[300],Y[300],O[300] ,mn, 'orangered')
    Orientation_ (X[350],Y[350],O[350] ,mn, 'orangered')
    Orientation_ (X[400],Y[400],O[400] ,mn, 'orangered')
    Orientation_ (X[450],Y[450],O[450] ,mn, 'orangered')
    Orientation_ (X[-1],Y[-1],O[-1], mn, 'orangered')
    
    
    
    plt.plot(X,Y,color='red',label='DOC (BL poids variable)')
    plt.plot(mm0[0],mm0[1],color='green',label='Trajectoire moyenne')
    plt.plot(mm0[6],mm0[7],color='green',linewidth=0.5,alpha = 0.5,label='Trajectoire individuelle')
    plt.plot(mm0[12],mm0[13],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[18],mm0[19],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[24],mm0[25],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[30],mm0[31],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[36],mm0[37],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[42],mm0[43],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[48],mm0[49],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[54],mm0[55],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[60],mm0[61],color='green',linewidth=0.5,alpha = 0.5)
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.grid()
    
    plt.ylabel("y [m]")
    plt.xlabel("x [m]")
    plt.legend()
    
    

def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 



def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r



n = 500
taux = 5/n
T = np.linspace(0,5,n)




theta=SX.sym('theta',n)
x=SX.sym('x',n)
y=SX.sym('y',n)


v1=SX.sym('v1',n)  

v2=SX.sym('v2',n) 

c = SX.sym('c',n)


dot_v1 = SX.zeros(1)
v1_prime = (v1[1:]-v1[:-1])/taux
for i in range (n-1):
    dot_v1 += c[i]*((v1_prime[i])**2)
    
A_forw = Function ('A_forw',[v1,c],[dot_v1])

dot_o = SX.zeros(1)
o_prime = (theta[1:]-theta[:-1])/taux
o_prime2 = (o_prime[1:]-o_prime[:-1])/taux
dot_o = SX.zeros(1)
for i in range (n-2):
    dot_o += c[i]*((o_prime2[i])**2)
    
A_ang = Function ('A_ang',[theta,c],[dot_o])

dot_v2 = SX.zeros(1)
v2_prime = (v2[1:]-v2[:-1])/taux
for i in range (n-1):
    dot_v2 += c[i]*((v2_prime[i])**2)
    
A_orth = Function ('A_orth',[v2,c],[dot_v2])


dot_v01 =  SX.zeros(1)
for i in range (n):
    dot_v01 += c[i]*((v1[i])**2)
    
V_forw = Function ('V_forw',[v1,c],[dot_v01])

dot_v02 =  SX.zeros(1)
for i in range (n):
    dot_v02 += c[i]*((v2[i])**2)
    
V_orth = Function ('V_orth',[v2,c],[dot_v02])

dot_o1 =  SX.zeros(1)
for i in range (n-1):
    dot_o1 += c[i]*((o_prime[i])**2)
    
V_ang = Function ('V_ang',[theta,c],[dot_o1])

x_prime = (x[1:]-x[:-1])/taux
x_prime2 = (x_prime[1:]-x_prime[:-1])/taux


y_prime = (y[1:]-y[:-1])/taux
y_prime2 = (y_prime[1:]-y_prime[:-1])/taux

ray = SX.zeros(1)

for i in range (n-2):
    ray += c[i]*(((x_prime[i]**2 + y_prime[i]**2 )**3 + 10**(-5))/((x_prime[i]* y_prime2[i] - x_prime2[i]* y_prime[i])**2 + 10**(-5)))
    
Courbure = Function('Courbure',[x,y,c],[ray])




options = {'maxfev': 100000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-10}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1], [0, 1], [0, 1] , [0, 1], [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1, 1, 1, 1,1], 1, 1)




def DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,X,Y,THETA,V1,V2):

    
    opti = casadi.Opti()   
    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    
    
    opti.minimize( A_forw(v1,c1* T**0) + A_ang(theta,c2 * T**0) + A_orth(v2,c3 * T**0) + V_forw(v1,c4 * T**0) + V_orth(v2,c5 * T**0) + V_ang(theta,c6 * T**0) + Courbure(x,y,c7 * T**0) )    
    
    opti.subject_to((Xi[0]- x[0])*0.2 == 0 )        
    opti.subject_to((Xi[1]- y[0])*0.2 == 0 )
    opti.subject_to(( Xi[2] - theta[0])*0.034 == 0 )    
    opti.subject_to( v1[0]*0.01 == 0 )
    opti.subject_to( v2[0]*0.01 == 0 )
    opti.subject_to( 0.034*(theta[1]-theta[0])/taux == 0 )
    

    opti.subject_to( x[1:]  == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )

    opti.subject_to( (x[-1]- Xf[0])*0.2 == 0 )
    opti.subject_to( (y[-1]- Xf[1])*0.2 == 0 )
    opti.subject_to( (theta[-1] - Xf[2])*0.034 == 0 )
    opti.subject_to( v1[-1]*0.01 == 0 ) 
    opti.subject_to( v2[-1]*0.01 == 0 )
    opti.subject_to( 0.034*(theta[-1]-theta[n-2])/taux == 0 )
    
    opti.set_initial(x, X)
    opti.set_initial(y, Y)
    opti.set_initial(theta, THETA)
    opti.set_initial(v1, V1)
    opti.set_initial(v2, V2)

    opti.solver('ipopt', {"expand" : True}, {"constr_viol_tol" : 0.01})      

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta),sol.value(v1),sol.value(v2)



def PDFO (C):
    
    c1,c2,c3,c4,c5,c6,c7 = C
    mk = 0
    
    print(c1,c2,c3,c4,c5,c6)
    
    
    if c1 < 0 : 
        mk += -c1
        c1 = 0
    if c2 < 0 : 
        mk += -c2
        c2 = 0
    if c3 < 0 : 
        mk += -c3
        c3 = 0
    if c4 < 0 : 
        mk += -c4
        c4 = 0
    if c5 < 0 : 
        mk += -c5
        c5 = 0
    if c6 < 0 : 
        mk += -c6
        c6 = 0
    if c7 < 0 : 
        mk += -c7
        c7 = 0
    
    x,y,o,v01,v02 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,Xmoy,Ymoy,Theta_moy,Vf,Vo)
    
    mm3 = (np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2+ np.linalg.norm(Theta_moy-o)**2)/n
    print(mm3)
        
    return float(mm3 + 10*mk)




DATA = ['N-0615.dat','S4015.dat','O0640.dat','E1540.dat']




T0 = np.loadtxt(DATA[3])
Xmoy = T0[0]  
Ymoy = T0[1]  
Theta_moy = T0[5]

Theta_moy[1]=Theta_moy[0]
Theta_moy[-1]=Theta_moy[n-2]

Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)




res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)




c1,c2,c3,c4,c5,c6,c7 = res.x




X,Y,O,v1,v2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,Xmoy,Ymoy,Theta_moy,Vf,Vo)


plt.figure(figsize = (40,7))    
plot_traj(DATA[3],X,Y,O)
plt.title(' Scénario {} \n  (RMSE(x,y) , RMSE(theta)) = ({} m , {} rad ) '.format(35, sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))
plt.savefig('RdC_PC_BL_avec_initialisation_scénario35.png') 