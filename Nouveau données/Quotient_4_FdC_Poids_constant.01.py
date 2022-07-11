#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *


n = 500
taux = 5/n
T = linspace(0,5,n)



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




x = SX.sym ('x', n)
y = SX.sym ('y', n)
theta = SX.sym ('theta', n)

xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)


M = SX.zeros(1)
Y = ((xf - x[1:])**2 + (yf - y[1:])**2 + (thetaf - theta[1:])**2 + 10**(-5) )/((xf - x[:-1])**2 + (yf - y[:-1])**2 + (thetaf - theta[:-1])**2 + 10**(-5) )
for i in range (Y.shape[0]):
    M += Y[i]
                                                
Direct = Function('Direct', [x,xf,y,yf,theta,thetaf],[M])

options = {'maxfev': 10000  , 'rhoend' : 1e-6}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1] , [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1,1], 1, 1)

def MH_PDFO (c):
    c1,c2,c3,c4 = c
    print(c)
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
    
    
    opti.minimize(  taux*( c1 *dot(u1,u1) + c2*dot(u2,u2 ) + c3*dot(u3 ,u3 ) +  c4 * Direct(y,Xf[1],x,Xf[0],theta,Xf[2]) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0 )
    opti.subject_to( w[0]  == 0 )
    opti.subject_to( v2[0] == 0 )
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
    opti.subject_to( x[-1]== Xf[0] )
    opti.subject_to( y[-1]== Xf[1] )
    opti.subject_to( theta[-1]== Xf[2] )
    opti.subject_to( v1[-1] == 0 ) 
    opti.subject_to( w[-1]  == 0 ) 
    opti.subject_to( v2[-1] == 0 )
    opti.subject_to( u1[-1] == 0 )
    opti.subject_to( u2[-1] == 0 )
    opti.subject_to( u3[-1] == 0 )
    
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()

    m01 = sqrt((np.linalg.norm(Xmoy-sol.value(x))**2 + np.linalg.norm(Ymoy-sol.value(y))**2 + np.linalg.norm(Theta_moy-sol.value(theta))**2)/n)
    m02 = 10* mk
    m03 = 10*(c1+c2+c3+c4-1)
    m1 = m01+m02+m03

    return float(m1)



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
    
    
    opti.minimize(  taux*( c1 *dot(u1,u1) + c2*dot(u2,u2 ) + c3*dot(u3 ,u3 ) +  c4 * Direct(y,Xf[1],x,Xf[0],theta,Xf[2]) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0 )
    opti.subject_to( w[0]  == 0 )
    opti.subject_to( v2[0] == 0 )
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
    opti.subject_to( x[-1]== Xf[0] )
    opti.subject_to( y[-1]== Xf[1] )
    opti.subject_to( theta[-1]== Xf[2] )
    opti.subject_to( v1[-1] == 0 ) 
    opti.subject_to( w[-1]  == 0 ) 
    opti.subject_to( v2[-1] == 0 )
    opti.subject_to( u1[-1] == 0 )
    opti.subject_to( u2[-1] == 0 )
    opti.subject_to( u3[-1] == 0 )
    
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta),sol.value(v1),sol.value(w),sol.value(v2),sol.value(u1),sol.value(u2),sol.value(u3)



T0 = np.loadtxt('E0640.dat')
Xmoy = T0[0]
Ymoy = T0[1]
Theta_moy = T0[5]
g = T0[4]
V1_moy = T0[2]*cos(Theta_moy)+T0[3]*sin(Theta_moy)
V2_moy = -T0[2]*sin(Theta_moy)+T0[3]*cos(Theta_moy)
Xi = [Xmoy[0], Ymoy[0],Theta_moy[0]]
Xf = [Xmoy[-1], Ymoy[-1],Theta_moy[-1]]




res = pdfo( MH_PDFO, [1/5,2/5,1/5,1/5],bounds=bounds1, constraints=lin_con1, options=options)

c1,c2,c3,c4 = res.x

x,y,o,v1,w,v2,u1,u2,u3 = MH_DOC(c1,c2,c3,c4,Xi,Xf)





plt.figure(figsize = (20,45))

plt.suptitle ("Xi = [%.4f,%.4f,%.4f]  Xf = [%.4f,%.4f,%.4f]  \n (alpha1, alpha2, alpha3, alpha4) = (%.2f,%.2f,%.2f,%.2f) " % (Xi[0],Xi[1],Xi[2]*(180/pi),Xf[0],Xf[1],Xf[2]*(180/pi),c1,c2,c3,c4) )

plt.subplot(4,2,1)
plt.title ("RMSE_plan = {} m ".format(sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 )/n)) )
plt.plot(x,y, label = 'DOC')
plt.plot(Xmoy, Ymoy, label = 'Donnée')
plt.plot(x[0],y[0],'*', label = 'Point initial')
plt.plot(x[-1],y[-1],'*', label = 'Point final')
#plt.ylim(-0.1,1.2)
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
tracer_orientation (Xmoy[0],Ymoy[0],Theta_moy[0] , 0.1, 0)
tracer_orientation (Xmoy[100],Ymoy[100],Theta_moy[100] , 0.1, 0)
tracer_orientation (Xmoy[200],Ymoy[200],Theta_moy[200] , 0.1, 0)
tracer_orientation (Xmoy[300],Ymoy[300],Theta_moy[300] , 0.1, 0)
tracer_orientation (Xmoy[400],Ymoy[400],Theta_moy[400] , 0.1, 0)

tracer_orientation (x[0],y[0],o[0] , 0.1, 1)
tracer_orientation (x[100],y[100],o[100] , 0.1, 0)
tracer_orientation (x[200],y[200],o[200] , 0.1, 0)
tracer_orientation (x[300],y[300],o[300] , 0.1, 0)
tracer_orientation (x[-1],y[-1],o[-1], 0.1, 0)
plt.legend()
plt.grid()
ax = plt.gca()
ax.set_aspect("equal")

plt.subplot(4,2,2)
plt.title ("coefficient de corrélation = {} ".format(corr2(y,Ymoy) ))
plt.plot(T,y, label = 'DOC')
plt.plot(T,Ymoy, label = 'Donné')
plt.xlabel("Times [s]")
plt.ylabel("Y [m]")
plt.legend()
plt.grid()


plt.subplot(4,2,3)
plt.title ("coefficient de corrélation = {} ".format(corr2(x,Xmoy) ))
plt.plot(T,x, label = 'DOC')
plt.plot(T,Xmoy, label = 'Donné')
plt.xlabel("Times [s]")
plt.ylabel("X [m]")
plt.legend()
plt.grid()

plt.subplot(4,2,4)
plt.title ("RMSE_ang = {} , coefficient de corrélation = {} ".format(sqrt((np.linalg.norm(Theta_moy-o)**2 )/n), corr2(o,Theta_moy) ))
plt.plot(T,o*(180/pi), label = "DOC")
plt.plot(T,Theta_moy*(180/pi), label = "Donné")
plt.xlabel("Times [s]")
plt.ylabel("Theta [°]")
plt.grid()
plt.legend()


plt.subplot(4,2,5)
plt.plot(T,v1)
plt.plot(T,V1_moy)

plt.xlabel("Times [s]")
plt.ylabel("V1 [m/s]")
plt.grid()

plt.subplot(4,2,6)
plt.plot(T,v2)
plt.plot(T,V2_moy)
plt.xlabel("Times [s]")
plt.ylabel("V2 [m/s]")
plt.grid()

plt.subplot(4,2,7)
plt.plot(T,w*(180/pi))
plt.xlabel("Times [s]")
plt.ylabel("W [°/s]")
plt.grid()

plt.subplot(4,2,8)
plt.plot(T[:-1],((Xf[0] - x[1:])**2 + (Xf[1] - y[1:])**2 )/((Xf[0]  - x[:-1])**2 + (Xf[1]- y[:-1])**2 + 0.01))
plt.xlabel("Times [s]")
plt.ylabel("f (x,y,yf,xf)")
plt.grid()

plt.savefig("essai.0.png")



sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 )/n), sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)



m0 = (Xmoy-x)**2 + (Ymoy-y)**2
m0 = sqrt(m0)
np.sum(m0)/n


m1 = (Theta_moy-o)**2
m1 = sqrt(m1)
np.sum(m1)/n





