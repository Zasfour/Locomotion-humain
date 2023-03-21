#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *




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




def orientation (x,y,theta, r, i):
    X00 = [x,x+r*cos(theta)]
    Y00 = [y,y+r*sin(theta)]
    
    
    if i == 1 :
        plt.plot(X00,Y00,'r',label='Orientation')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'red' )
        plt.legend()
    else :
        plt.plot(X00,Y00,'r')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'red' )
        


m01 ,m02 = 216.6058685341806 ,18770.730081753416
m11 ,m12 = 0.0 ,91445082.23389618
m21 ,m22 = 285.12944960426574 ,10778.161111878238
m31 ,m32 = 486.3454140114936 ,488.23333202168567
m41 ,m42 = 4.249648425966459e-05 ,0.6546337152297712
m51 ,m52 = 97.61305051332354 ,264.71837454603855
m61 ,m62 = 0.0 ,9144.508223389617


n = 500
taux = 5/n
T = linspace(0,5,n)




x = SX.sym('x',n)
y = SX.sym('y',n)
xf = SX.sym('xf',1)
yf = SX.sym('yf',1)


N = SX.zeros(1)
Y = ((xf - x[1:])**2 + (yf - y[1:])**2 + 10**(-5) )/((xf - x[:-1])**2 + (yf - y[:-1])**2 + 10**(-5) )
for i in range (Y.shape[0]):
    N += Y[i]
                                                
Direct = Function('Direct', [x,xf,y,yf],[N])

M = []
for i in range (1):
    M = vertcat(M,fmax(0,(x[i+1]-xf)**2 + (y[i+1]-yf)**2 - ((x[i]-xf)**2 + (y[i]-yf)**2)))
N0 = SX.zeros(1)
for i in range (M.shape[0]):
    N0 += M[i]
    
Max = Function ('Max',[x,y,xf,yf],[N0])


def MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf):
    
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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 *((Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41))+ c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
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



T0 = np.loadtxt('O1515.dat')
Xmoy = T0[0]
Ymoy = T0[1]
Theta_moy = T0[5]
g = T0[4]
V1_moy = T0[2]*cos(Theta_moy)+T0[3]*sin(Theta_moy)
V2_moy = -T0[2]*sin(Theta_moy)+T0[3]*cos(Theta_moy)
W_moy = (vertcat(0,Theta_moy[1:])-Theta_moy)/taux
Xi = [Xmoy[0], Ymoy[0],Theta_moy[0]]
Xf = [Xmoy[-1], Ymoy[-1],Theta_moy[-1]]


c1 = 0.25
c2 = 0.15
c3 = 0.35
c4 = 0.05
c5 = 0.01
c6 = 0.01
c7 = 0.18
x,y,o,v1,w,v2,u1,u2,u3 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)



plt.figure(figsize = (20,45))

plt.suptitle ("Xi = [%.4f,%.4f,%.4f]  Xf = [%.4f,%.4f,%.4f]  \n (alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7) = (%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) " % (Xi[0],Xi[1],Xi[2]*(180/pi),Xf[0],Xf[1],Xf[2]*(180/pi),c1,c2,c3,c4,c5,c6,c7) )

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
#plt.title ("Coefficient de corrélation = {} ".format( corr2(v1,V1_moy) ))
plt.plot(T,v1)
plt.plot(T,V1_moy)
plt.xlabel("Times [s]")
plt.ylabel("V1 [m/s]")
plt.grid()

plt.subplot(4,2,6)
#plt.title ("Coefficient de corrélation = {} ".format( corr2(v2,V2_moy) ))
plt.plot(T,v2)
plt.plot(T,V2_moy)
plt.xlabel("Times [s]")
plt.ylabel("V2 [m/s]")
plt.grid()

plt.subplot(4,2,7)
#plt.title ("Coefficient de corrélation = {} ".format( corr2(w,W_moy) ))
plt.plot(T,w*(180/pi))
plt.plot(T,W_moy*(180/pi))
plt.xlabel("Times [s]")
plt.ylabel("W [°/s]")
plt.grid()

plt.subplot(4,2,8)
plt.plot(T[:-1],((Xf[0] - x[1:])**2 + (Xf[1] - y[1:])**2 )/((Xf[0]  - x[:-1])**2 + (Xf[1]- y[:-1])**2 + 0.01))
plt.xlabel("Times [s]")
plt.ylabel("f (x,y,yf,xf)")
plt.grid()
plt.savefig("essaye.png")




m0 = (Xmoy-x)**2 + (Ymoy-y)**2
m0 = sqrt(m0)
np.sum(m0)/n




m1 = (Theta_moy-o)**2
m1 = sqrt(m1)
np.sum(m1)/n