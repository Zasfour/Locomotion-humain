import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *

def Orientation_ (x,y,theta, m, couleur ):
    plt.arrow(x, y, (m/2)*cos(theta),(m/2)*sin(theta), color = couleur, head_width = m/2, head_length = m)



def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r

Groupe2 = ['O0615.dat','O1515.dat','N0615.dat','S0615.dat','S1515.dat','E0615.dat','N1515.dat','E1515.dat','O-0615.dat','N-0615.dat','S-0615.dat','E-0615.dat']


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


Groupe1 = ['S1500.dat','E1500.dat','O1500.dat','N1500.dat']



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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * ((Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41))+ c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
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



options = {'maxfev': 1000000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1], [0, 1], [0, 1] , [0, 1], [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1, 1, 1, 1 ,1], 1, 1)




def PDFO (C):
    c1,c2,c3,c4,c5,c6,c7 = C
    print(C)
    mk = 0
    
    if c1 < 0:
        c1 = -c1
        mk+= c1
    if c2 < 0:
        c2 = -c2
        mk+= c2
    if c3 < 0:
        c3 = -c3
        mk+= c3
    if c4 < 0:
        c4 = -c4
        mk+= c4
    if c5 < 0:
        c5 = -c5
        mk+= c5
    if c6 < 0:
        c6 = -c6
        mk+= c6
    if c7 < 0:
        c7 = -c7
        mk+= c7
        
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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * ((Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41)) + c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
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
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    m001 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    m002 = 10* np.abs (c1 + c2 + c3 + c4 + c5 + c6 + c7 -1)
    m003 = 10* mk
    
    m1 = float(m001+m002+m003)

    return m1

color = ['orangered', 'olive']


for j in range (len(Groupe2)):

    T0 = np.loadtxt(Groupe2[j])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = T0[5]
    Xi = [Xmoy[0], Ymoy[0],Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1],Theta_moy[-1]]

    res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)



    c1,c2,c3,c4,c5,c6,c7 = res.x

    x,y,o,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)


    plt.figure(figsize=(15,15))
    
    m = (((x[0]-x[-1])**2 + (y[0]-y[-1])**2 )/n)
    m1 = (((Xmoy[0]-Xmoy[-1])**2 + (Ymoy[0]-Ymoy[-1])**2 )/n)

    if m1 < 1 :
        m = m*15
        m1 = m1*15

    Orientation_ (x[0],y[0],o[0] ,m, color[0])
    Orientation_  (x[100],y[100],o[100] ,m,color[0])
    Orientation_ (x[200],y[200],o[200] ,m, color[0])
    Orientation_ (x[300],y[300],o[300] ,m, color[0])
    Orientation_ (x[400],y[400],o[400] ,m, color[0])

    Orientation_ (x[-1],y[-1],o[-1], m, color[0])

    Orientation_ (Xmoy[0],Ymoy[0],Theta_moy[0] ,m1, color[1])
    Orientation_ (Xmoy[100],Ymoy[100],Theta_moy[100] ,m1, color[1])
    Orientation_ (Xmoy[200],Ymoy[200],Theta_moy[200] ,m1, color[1])
    Orientation_ (Xmoy[300],Ymoy[300],Theta_moy[300] ,m1, color[1])
    Orientation_ (Xmoy[400],Ymoy[400],Theta_moy[400] ,m1, color[1])
    Orientation_ (Xmoy[-1],Ymoy[-1],Theta_moy[-1] ,m1, color[1])



    plt.plot(x,y,'orange', label = 'Bi-level by (PDFO)')
    plt.plot(Xmoy,Ymoy,'greenyellow', label = 'mean trajectory by {}'.format(Groupe2[j]))

    plt.plot(x[0],y[0],'o', color = 'tab:green', label = 'initial position')
    plt.plot(x[-1],y[-1],'o', color = 'darkorange', label = 'goal position')
    plt.title("[RMSE_plan, RMSE_angular] = [{} m , {} rad]".format(sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2  )/n), sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)))
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlabel ("X [m]")
    plt.ylabel ("Y [m]")
    plt.savefig("PDFO_{}.png".format(Groupe2[j]))