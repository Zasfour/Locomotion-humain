#!/usr/bin/env python
# coding: utf-8


import cyipopt
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from scipy.optimize import approx_fprime



n = 500
T = 5
time = np.linspace(0,T,n)
taux = T/n

m01 ,m02 = 62.60648207785031 ,8716.76547379362
m11 ,m12 = 0.0 ,1154653828.5970802
m21 ,m22 = 20.543893994759976 ,7861.558309796407
m31 ,m32 = 488.0082137605221 ,489.1307329027765
m41 ,m42 = 0.003127686679216384 ,0.15123248841357761
m51 ,m52 = 41.753853472012764 ,73.75947511558543
m61 ,m62 = 0.0 ,115465.38285970801


Groupe1 = ['E1500.dat','S1500.dat','O1500.dat','N1500.dat']

def Orientation_ (x,y,theta, m, couleur ):
    plt.arrow(x, y, (m/2)*cos(theta),(m/2)*sin(theta), color = couleur, head_width = m/2, head_length = m)


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
for i in range (n-1):
    M = vertcat(M,fmax(0,(x[i+1]-xf)**2 + (y[i+1]-yf)**2 - ((x[i]-xf)**2 + (y[i]-yf)**2)))

N0 = SX.zeros(1)
for i in range (M.shape[0]):
    N0 += M[i]
    
Max = Function ('Max',[x,y,xf,yf],[N0])


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)

xi = SX.sym('xi',1)
yi = SX.sym('yi',1)
thetai = SX.sym('thetai',1)


## Position
x=SX.sym('x',n)

y=SX.sym('y',n)

theta=SX.sym('theta',n)

X0=SX.sym('X0',n)

Y0=SX.sym('Y',n)

Theta=SX.sym('Theta',n)

## Vitesse
v1=SX.sym('v1',n)  

v2=SX.sym('v2',n)  

w=SX.sym('w',n)

## accélération 
u1 = SX.sym('u1',n)
u2 = SX.sym('u2',n)
u3 = SX.sym('u3',n)


## Notre fonction objectif
Objectif = SX.zeros(1)
Objectif = dot(x-X0,x-X0) + dot(y-Y0,y-Y0) + dot(theta-Theta,theta-Theta)    

F_obj = Function('F_obj',[x,y,theta,X0,Y0,Theta],[Objectif])

## gradient de la fonction objectif 
X = vertcat(x,y,theta,v1,w,v2,u1,u2,u3)
H_obj , J_obj = hessian(Objectif,X) 
Gradient_obj = Function("Gradient_obj",[u1,u2,u3,v1,w,v2,x,y,theta,X0,Y0,Theta],[J_obj])


## Contrainte 
G0 = SX.sym('G0',9)
G0[0] = x[0] - xi
G0[1] = y[0] - yi
G0[2] = theta[0] - thetai
G0[3] = v1[0]
G0[4] = w[0]
G0[5] = v2[0]
G0[6] = u1[0]
G0[7] = u2[0]
G0[8] = u3[0]
G1 = SX.sym('G1',6*(n-1))
for i in range (n-1) :
    G1[6*i] = x[i] + taux * (v1[i] * cos(theta[i]) - sin(theta[i])*v2[i])- x[i+1]
    G1[6*i+1] = y[i] + taux *(v1[i] * sin(theta[i]) + cos(theta[i])*v2[i]) - y[i+1]
    G1[6*i+2] = theta[i] + taux * w[i]  - theta[i+1]
    G1[6*i+3] = v1[i] + taux * u1[i]  - v1[i+1]
    G1[6*i+4] = w[i] + taux * u2[i]  - w[i+1]
    G1[6*i+5] = v2[i] + taux * u3[i]  - v2[i+1]
G2 = SX.sym('G2',9)
G2[0] = x[-1] - xf
G2[1] = y[-1] - yf
G2[2] = theta[-1] - thetaf
G2[3] = v1[-1]
G2[4] = w[-1]
G2[5] = v2[-1]
G2[6] = u1[-1]
G2[7] = u2[-1]
G2[8] = u3[-1]
G = vertcat(G0,G1,G2)  ### première partie des contrainte (il ne reste que le déterminant)
Contraint = Function("contrainte",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[G])


F_K = SX.sym('F_K',7)
F_K[0] = ((dot(u1,u1)-m01 )/(m02-m01))
F_K[1] = ((dot(u2,u2 )-m11 )/(m12-m11))
F_K[2] = ((dot(u3 ,u3)-m21)/(m22-m21)) 
F_K[3] = ((Direct(y,yf,x,xf)-m31 )/(m32-m31))
F_K[4] = ((Max(x,y,xf,yf)-m41 )/(m42-m41))
F_K[5] = (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51))
F_K[6] = ((dot(w,w)-m61)/(m62-m61))

F_G = vertcat(F_K ,G0, G1,G2)
J = SX.zeros(X.shape[0],F_G.shape[0])
J_lambda = SX.zeros(X.shape[0],G.shape[0])



for i in range (F_G.shape[0]):
    HJ,J[:,i] = hessian(F_G[i],X)
        
for i in range (G.shape[0]):
    HJ,J_lambda[:,i] = hessian(G[i],X)
    
    
Lambda = SX.sym('Lambda',6*(n+2))
c1 = SX.sym('c1',1)
c2 = SX.sym('c2',1)
c3 = SX.sym('c3',1)
c4 = SX.sym('c4',1)
c5 = SX.sym('c5',1)
c6 = SX.sym('c6',1)
c7 = SX.sym('c7',1)

C = vertcat(c1,c2,c3,c4,c5,c6,c7,Lambda)


res_ = SX.zeros(1)

res_ = c1*((dot(u1,u1)-m01 )/(m02-m01)) + c2*((dot(u2,u2 )-m11 )/(m12-m11)) + c3*((dot(u3 ,u3)-m21)/(m22-m21))  + c4*((Direct(y,yf,x,xf)-m31 )/(m32-m31)) + c5*((Max(x,y,xf,yf)-m41 )/(m42-m41)) + c6*(((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7*((dot(w,w)-m61)/(m62-m61)) 

for i in range (6*(n+2)):
    res_ += G[i]*Lambda[i]
    
UNN , res = hessian(res_,X)
    
R = Function('R',[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf,c1,c2,c3,c4,c5,c6,c7,Lambda],[dot(res,res)])


Res = np.matmul(J,C)
Residu = Function('Residu',[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf,c1,c2,c3,c4,c5,c6,c7,Lambda],[dot(Res,Res)])

Matrice_J = Function("Matrice_J",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[J])      ## pour le calcule du déterminant 
Matrice_J_lambda = Function("Matrice_J_lambda",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[J_lambda])      ## pour le calcule du déterminant 



####### Jacobien CONTRAINTE 
Jacobien_contrainte = SX.zeros(X.shape[0]*G.shape[0])
for i in range (G.shape[0]):
    H_cont , Jacobien_contrainte[i*9*n:(i+1)*9*n] = hessian(G[i],X)
                                                        ### gradient des contrainte sauf celle du déterminant 

Jac_contrainte = Function("Jac_contrainte",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[Jacobien_contrainte])  ### ajouter après la partie du déterminant 


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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * (Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41)+ c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
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


def KKT (X,Y,Theta,V1,W,V2,U1,U2,U3,Xi,Xf):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    c1 = opti.variable()
    c2 = opti.variable()
    c3 = opti.variable()
    c4 = opti.variable()
    c5 = opti.variable()
    c6 = opti.variable()
    c7 = opti.variable()
    Lambda = opti.variable(6*(n+2))



    opti.minimize( R(U1,U2,U3,V1,W,V2,X,Y,Theta, Xi[0],Xi[1] ,Xi[2] , Xf[0] ,Xf[1],Xf[2], c1,c2,c3,c4,c5,c6,c7,Lambda)) 
    opti.subject_to( 0 <= c1)
    opti.subject_to( 0 <= c2 )
    opti.subject_to( 0 <= c3 )
    opti.subject_to( 0 <= c4 )
    opti.subject_to( 0 <= c5 )
    opti.subject_to( 0 <= c6 )
    opti.subject_to( 0 <= c7 )
    opti.subject_to(  c1 + c2 + c3 + c4 + c5 + c6 +c7  == 1)

    opti.solver('ipopt', {"expand" : True}, {"tol" : 10**(-25)}  ) 
    #opti.solver('ipopt')             
    
 
    sol = opti.solve()
    
    return sol.value (c1),sol.value (c2),sol.value (c3),sol.value (c4),sol.value (c5),sol.value (c6) ,sol.value (c7)
    

color = ['orangered', 'olive']


for j in range (len(Groupe1)):

    T0 = np.loadtxt(Groupe1[j])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = T0[5]
    V1_moy = T0[2]*cos(Theta_moy)+T0[3]*sin(Theta_moy)
    V2_moy = -T0[2]*sin(Theta_moy)+T0[3]*cos(Theta_moy)
    W_moy = (vertcat(Theta_moy[1:],Theta_moy[-1])-Theta_moy)/taux
    Xi = [Xmoy[0], Ymoy[0],Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1],Theta_moy[-1]]

    U1_moy = (vertcat(V1_moy[1:],V1_moy[-1])-V1_moy)/taux
    U2_moy = (vertcat(W_moy[1:],W_moy[-1])-W_moy)/taux
    U3_moy = (vertcat(V2_moy[1:],V2_moy[-1])-V2_moy)/taux


    c1,c2,c3,c4,c5,c6,c7 = KKT (Xmoy,Ymoy,Theta_moy,V1_moy,W_moy,V2_moy,U1_moy,U2_moy,U3_moy,Xi,Xf)


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



    plt.plot(x,y,'orange', label = 'IKKT')
    plt.plot(Xmoy,Ymoy,'greenyellow', label = 'mean trajectory by {}'.format(Groupe1[j]))

    plt.plot(x[0],y[0],'o', color = 'tab:green', label = 'initial position')
    plt.plot(x[-1],y[-1],'o', color = 'darkorange', label = 'goal position')
    plt.title("[RMSE_plan, RMSE_angular] = [{} m , {} rad]".format(sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2  )/n), sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)))
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlabel ("X [m]")
    plt.ylabel ("Y [m]")
    plt.savefig("KKT_{}.png".format(Groupe1[j]))