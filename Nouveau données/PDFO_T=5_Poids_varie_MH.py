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
T= linspace(0,1,n)


def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )



def DOC_MH (c1,c2,c3):
    
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


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *u3 ,u3 ) ) )    # ma fonction objetion

        # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == Xmoy[0] + 10**(-4))       
    opti.subject_to( y[0] == Ymoy[0] + 10**(-4))    
    opti.subject_to( theta[0] == Theta_moy[0] + 10**(-4))


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
    opti.subject_to( x[-1]==Xmoy[-1] + 10**(-4))
    opti.subject_to( y[-1]==Ymoy[-1] + 10**(-4))
    opti.subject_to( theta[-1]==Theta_moy[-1] + 10**(-4))


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    return sol.value(x), sol.value(y), sol.value(theta)



############################################  KKT 

xi = SX.sym('xi',1)                   
yi = SX.sym('yi',1)                
thetai = SX.sym('thetai',1)


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)


A = SX.sym('A',6)
B = SX.sym('B',6)
C = SX.sym('C',6)

## Position
x=SX.sym('x',n)
x_prime = SX.sym('x_prime', n+1)
x_prime[0] = x[0]
x_prime[1:] =x


y=SX.sym('y',n)
y_prime = SX.sym('y_prime', n+1)
y_prime[0] = y[0]
y_prime[1:] =y

theta=SX.sym('theta',n)
theta_prime = SX.sym('theta_prime', n+1)
theta_prime[0] = theta[0]
theta_prime[1:] =theta


## Vitesse
v1=SX.sym('v1',n)  
v1_prime = SX.sym('v1_prime', n+1)
v1_prime[0] = 0
v1_prime[n] = 0
v1_prime[1:n] =v1[0:n-1]

v1_prime_1 = SX.sym('v1_prime_1', n+1)
v1_prime_1[0] = v1[0]
v1_prime_1[1:] =v1


v2=SX.sym('v2',n)  
v2_prime = SX.sym('v2_prime', n+1)
v2_prime[0] = 0
v2_prime[n] = 0
v2_prime[1:n] =v2[0:n-1]

v2_prime_1 = SX.sym('v2_prime_1', n+1)
v2_prime_1[0] = v2[0]
v2_prime_1[1:] =v2


w=SX.sym('w',n)  
w_prime = SX.sym('w_prime', n+1)
w_prime[0] = 0
w_prime[n] = 0
w_prime[1:n] =w[0:n-1]

w_prime_1 = SX.sym('w_prime_1', n+1)
w_prime_1[0] = w[0]
w_prime_1[1:] =w


## Accélération 

u1=SX.sym('u1',n)  
u1_prime = SX.sym('u1_prime', n+1)
u1_prime[0] = 0
u1_prime[n] = 0
u1_prime[1:n] = u1[0:n-1]

u2=SX.sym('u2',n)  
u2_prime = SX.sym('u2_prime', n+1)
u2_prime[0] = 0
u2_prime[n] = 0
u2_prime[1:n] = u2[0:n-1]

u3=SX.sym('u3',n)  
u3_prime = SX.sym('u3_prime', n+1)
u3_prime[0] = 0
u3_prime[n] = 0
u3_prime[1:n] = u3[0:n-1]

Lambda = SX.sym('Lambda',n+3, 6)
c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 
c3 = C[0]* (T**0) + C[1]* (T**1) + C[2]* (T**2) + C[3]* (T**3) + C[4]* (T**4) + C[5]* (T**5) 
p1=vertcat(xi + 10**(-4),x_prime[2:] + 10**(-4),xf + 10**(-4))   
h= Function('h',[x, xi, xf],[p1])

p2=vertcat(0, v1)   
K = Function('K', [v1], [p2])

p =vertcat(v1[1:],0)
g = Function ('g',[v1],[p])
Y1_K = (x_prime+taux*(v1_prime*cos(theta_prime) - v2_prime*sin(theta_prime)) - h(x, xi,xf))
Y2_K = (y_prime+taux*(v1_prime*sin(theta_prime) + v2_prime*cos(theta_prime)) - h(y, yi,yf)) 
Y3_K = (theta_prime+taux*w_prime - h(theta, thetai,thetaf))

U1 = (g(v1) + 10**(-4) -v1)/taux - u1
U2 = (g(w) + 10**(-4) -w)/taux  - u2
U3 = (g(v2) + 10**(-4) -v2)/taux  - u3 

Y4_K = K(U1) 
Y5_K = K(U2)
Y6_K = K(U3)
Y_K = SX.sym('Y_K',n+1 , 6)        ## notre contrainte

for i in range (0,n+1):
    Y_K[i,0]= Y1_K[i]
    Y_K[i,1]= Y2_K[i]
    Y_K[i,2]= Y3_K[i]       
    Y_K[i,3]= Y4_K[i]       
    Y_K[i,4]= Y5_K[i]       
    Y_K[i,5]= Y6_K[i]       
    
## notre terme qui est relié a la contrainte.
G_lambda = 0

for i in range (n+1):
    G_lambda += dot(Y_K[i,:], Lambda[i,:])
    
G_lambda += (v1[0]-0.0001)*Lambda[n+1,0] + (w[0]-0.0001)*Lambda[n+1,1] + (v2[0]-0.0001)*Lambda[n+1,2] 
G_lambda += (v1[-1]-0.0001)*Lambda[n+1,3] + (w[-1]-0.0001)*Lambda[n+1,4] + (v2[-1]-0.0001)*Lambda[n+1,5] 

G_lambda += (u1[0]-0.0001)*Lambda[n+2,0] + (u2[0]-0.0001)*Lambda[n+2,1] + (u3[0]-0.0001)*Lambda[n+2,2] 
G_lambda += (u1[-1]-0.0001)*Lambda[n+2,3] + (u2[-1]-0.0001)*Lambda[n+2,4] + (u3[-1]-0.0001)*Lambda[n+2,5] 


F_val_K =  taux*(  dot(c1 *u1,u1) +  dot(c2 *u2,u2) +  dot(c3*u3,u3))

## le Lagrangien 
L_val_K = F_val_K + G_lambda
grad_L_K = SX.zeros(9, n)
for i in range (n):
    grad_L_K[0,i]= jacobian(L_val_K, v1[i])
    grad_L_K[1,i]= jacobian(L_val_K, w[i])
    grad_L_K[2,i]= jacobian(L_val_K, v2[i])
    grad_L_K[3,i]= jacobian(L_val_K, x[i])
    grad_L_K[4,i]= jacobian(L_val_K, y[i])
    grad_L_K[5,i]= jacobian(L_val_K, theta[i])
    grad_L_K[6,i]= jacobian(L_val_K, u1[i])
    grad_L_K[7,i]= jacobian(L_val_K, u2[i])
    grad_L_K[8,i]= jacobian(L_val_K, u3[i])
    
    
    
R_K = Function ('R_K', [u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A, B, C ,xi,yi,thetai, xf,yf,thetaf  ], [dot(grad_L_K,grad_L_K)])
    
def MH_KKT (X,Y,Theta,V1,W,V2,U1,U2,U3):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème


    A = opti.variable(6)
    B = opti.variable(6)
    C = opti.variable(6)
    c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 
    c3 = C[0]* (T**0) + C[1]* (T**1) + C[2]* (T**2) + C[3]* (T**3) + C[4]* (T**4) + C[5]* (T**5) 

    Lambda = opti.variable(n+3,6)


    opti.minimize( R_K(U1, U2, U3,V1,W,V2,X,Y,Theta, Lambda, A,B,C,  X[0],Y[0],Theta[0], X[-1],Y[-1],Theta[-1] )) 
    
    for j in range(n):

        opti.subject_to( 0 <= c1[j] )

        opti.subject_to( 0 <= c2[j] )

        opti.subject_to( 0 <= c3[j] )

        opti.subject_to(  c1[j] + c2[j] + c3[j] == 1)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    return sol.value(A),sol.value(B),sol.value(C)




##################################### BL1

X1=SX.sym('X1',n)
X2=SX.sym('X2',n)  
X3=SX.sym('X3',n)  


m = SX.sym('m',1)
m = (dot(X1-x,X1-x) + dot(X2-y,X2-y) + dot(X3-theta,X3-theta))

M = Function ('M', [x,y,theta, X1,X2,X3], [m])
def MH_BL1 (X,Y,THETA,V1,W,V2,U1,U2,U3):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)  
    v2 = opti.variable(n)        
    w = opti.variable(n)         


    ## les accélération 
    u1 = opti.variable(n)        
    u3 = opti.variable(n)       
    u2 = opti.variable(n)       

    A = opti.variable(6)
    B = opti.variable(6)
    C = opti.variable(6)
    
    c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 
    c3 = C[0]* (T**0) + C[1]* (T**1) + C[2]* (T**2) + C[3]* (T**3) + C[4]* (T**4) + C[5]* (T**5) 

    Lambda = opti.variable(n+3,6)


    opti.minimize( 5*10**(2) *  R_K(u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A,B,C,  X[0],Y[0],THETA[0], X[-1],Y[-1],THETA[-1]) + M(x,y,theta, X,Y,THETA) )
    #opti.minimize( M(x,y,theta, X,Y,THETA) ) 
    
    
    for j in range(n):
        opti.subject_to( 10**(-7) <= c1[j] )

        opti.subject_to( 10**(-7) <= c2[j] )

        opti.subject_to( 10**(-7) <= c3[j] )

        opti.subject_to(  c1[j] + c2[j] + c3[j] == 1)
        
    opti.subject_to( x[0] == X[0] + 10**(-4))        
    opti.subject_to( y[0] == Y[0] + 10**(-4))
    opti.subject_to( theta[0] == THETA[0] + 10**(-4))
    
    opti.subject_to( v1[0] == 0.0001 )
    opti.subject_to( w[0]  == 0.0001 )
    opti.subject_to( v2[0] == 0.0001 )
    opti.subject_to( v1[-1] == 0.0001 )
    opti.subject_to( w[-1]  == 0.0001 )
    opti.subject_to( v2[-1] == 0.0001 )
        
    opti.subject_to( u1[-1] == 0.0001 )
    opti.subject_to( u2[-1] == 0.0001 )
    opti.subject_to( u3[-1] == 0.0001 )
    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u3[0] == 0.0001 )



    ## pour les contraintes d'égaliter
    #opti.subject_to(  R_K(u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A,B,C,  X[0],Y[0],THETA[0], X[-1],Y[-1],THETA[-1]) <= 10**(-2))
    opti.subject_to( x[1:] + 10**(-4) == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] + 10**(-4) == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]  + 10**(-4) == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] + 10**(-4))  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] + 10**(-4)) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] + 10**(-4)) )


        ## pour les conditions finales
    opti.subject_to( x[-1]==X[-1] + 10**(-4))
    opti.subject_to( y[-1]==Y[-1] + 10**(-4))
    opti.subject_to( theta[-1]==THETA[-1] + 10**(-4))
    
    
    opti.set_initial(u1, U1)
    opti.set_initial(u2, U2)
    opti.set_initial(u3, U3)
    opti.set_initial(v1, V1)
    opti.set_initial(w, W)
    opti.set_initial(v2, V2)
    opti.set_initial(x, X)
    opti.set_initial(y, Y)
    opti.set_initial(theta, THETA)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    return sol.value(A),sol.value(B),sol.value(C), sol.value(x), sol.value(y), sol.value(theta)




################################################ PDFO


options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0, 0,0,0,0,0], 0, 1))    
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0,0, 0,0,0,0,0], 0, 1))    
def MH_PDFO (C):
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    c3 = C0* (T**0) + C1* (T**1) + C2* (T**2) + C3* (T**3) + C4* (T**4) + C5* (T**5)
    
    print(C)
    
    c01 = c1
    c02 = c2
    c03 = c3
    
    
    mk = 0
    
    for j in range (c1.shape[0]):
        if c1[j] < 0 :
            c01[j] = - c1[j] 
            mk = mk - c1[j] 
        if c2[j] < 0 :
            c02[j] = - c2[j]
            mk = mk - c2[j] 
        if c3[j] < 0 :
            c03[j] = - c3[j]
            mk = mk - c3[j] 
            
            
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


    opti.minimize(  taux*( dot(c01 *u1,u1) +  dot(c02 *u2,u2 ) + dot(c03 *u3 ,u3 ) ) )    # ma fonction objetion

        # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == Xmoy[0] + 10**(-4))       
    opti.subject_to( y[0] == Ymoy[0] + 10**(-4))    
    opti.subject_to( theta[0] == Theta_moy[0] + 10**(-4))


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
    opti.subject_to( x[-1]==Xmoy[-1] + 10**(-4))
    opti.subject_to( y[-1]==Ymoy[-1] + 10**(-4))
    opti.subject_to( theta[-1]==Theta_moy[-1] + 10**(-4))


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    
    m01 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    
    m02 = 10*abs(np.sum(c1 + c2 + c3) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return float(m1)



KKT_RMSE_PLAN = np.zeros(23)
KKT_RMSE_ang_rad = np.zeros(23)
KKT_RMSE_ang_degree = np.zeros(23)



for i in range (23):
    T0 = np.loadtxt(data2[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = atan(T0[3]/T0[2])

    M00 = vertcat(0,Theta_moy[1:])

    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    V1_moy = cos(Theta_moy)*T0[2]+T0[3]*sin(Theta_moy)

    V2_moy = -T0[2]*sin(Theta_moy) + cos(Theta_moy)*T0[3]

    W_moy  = (M00-T0[5])/taux

    M1 = vertcat(V1_moy[1:],0)
    M2 = vertcat(W_moy[1:],0)
    M3 = vertcat(V2_moy[1:],0)

    U1_moy = (M1-V1_moy)/taux
    U2_moy = (M2-W_moy)/taux
    U3_moy = (M3-V2_moy)/taux
    
    A_KKT,B_KKT,C_KKT = MH_KKT (Xmoy,Ymoy,Theta_moy,V1_moy,W_moy,V2_moy,U1_moy,U2_moy,U3_moy)
    
    c1_KKT = A_KKT[0] + A_KKT[1]*T + A_KKT[2]* (T**2) + A_KKT[3]* (T**3) + A_KKT[4]* (T**4) + A_KKT[5]* (T**5) 
    c2_KKT = B_KKT[0] + B_KKT[1]*T + B_KKT[2]* (T**2) + B_KKT[3]* (T**3) + B_KKT[4]* (T**4) + B_KKT[5]* (T**5) 
    c3_KKT = C_KKT[0] + C_KKT[1]*T + C_KKT[2]* (T**2) + C_KKT[3]* (T**3) + C_KKT[4]* (T**4) + C_KKT[5]* (T**5)
    
    X_KKT , Y_KKT, Theta_KKT = DOC_MH (c1_KKT,c2_KKT,c3_KKT)
    
    KKT_RMSE_PLAN[i] = sqrt((np.linalg.norm(Xmoy-X_KKT)**2 + np.linalg.norm(Ymoy-Y_KKT)**2 )/n)
    KKT_RMSE_ang_rad[i] = sqrt((np.linalg.norm(Theta_moy-Theta_KKT)**2 )/n)
    KKT_RMSE_ang_degree[i] = sqrt((np.linalg.norm(Theta_moy-Theta_KKT)**2 )/n) * (180/pi)   




df = pd.DataFrame({'Mean_traj (holonomique model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : KKT_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : KKT_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : KKT_RMSE_ang_degree})

df




df.to_csv('RMSE_poids_variable_KKT_MH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_KKT_MH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_KKT_MH_T=5.png')




BL1_RMSE_PLAN = np.zeros(23)
BL1_RMSE_ang_rad = np.zeros(23)
BL1_RMSE_ang_degree = np.zeros(23)



for i in range (23):
    T0 = np.loadtxt(data2[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = atan(T0[3]/T0[2])

    M00 = vertcat(0,Theta_moy[1:])

    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    V1_moy = cos(Theta_moy)*T0[2]+T0[3]*sin(Theta_moy)

    V2_moy = -T0[2]*sin(Theta_moy) + cos(Theta_moy)*T0[3]

    W_moy  = (M00-T0[5])/taux

    M1 = vertcat(V1_moy[1:],0)
    M2 = vertcat(W_moy[1:],0)
    M3 = vertcat(V2_moy[1:],0)

    U1_moy = (M1-V1_moy)/taux
    U2_moy = (M2-W_moy)/taux
    U3_moy = (M3-V2_moy)/taux
    
    A_KKT,B_KKT,C_KKT = MH_KKT (Xmoy,Ymoy,Theta_moy,V1_moy,W_moy,V2_moy,U1_moy,U2_moy,U3_moy)
    
    c1_KKT = A_KKT[0] + A_KKT[1]*T + A_KKT[2]* (T**2) + A_KKT[3]* (T**3) + A_KKT[4]* (T**4) + A_KKT[5]* (T**5) 
    c2_KKT = B_KKT[0] + B_KKT[1]*T + B_KKT[2]* (T**2) + B_KKT[3]* (T**3) + B_KKT[4]* (T**4) + B_KKT[5]* (T**5) 
    c3_KKT = C_KKT[0] + C_KKT[1]*T + C_KKT[2]* (T**2) + C_KKT[3]* (T**3) + C_KKT[4]* (T**4) + C_KKT[5]* (T**5)
    
    X_KKT , Y_KKT, Theta_KKT = DOC_MH (c1_KKT,c2_KKT,c3_KKT)
    
    BL1_RMSE_PLAN[i] = sqrt((np.linalg.norm(Xmoy-X_KKT)**2 + np.linalg.norm(Ymoy-Y_KKT)**2 )/n)
    BL1_RMSE_ang_rad[i] = sqrt((np.linalg.norm(Theta_moy-Theta_KKT)**2 )/n)
    BL1_RMSE_ang_degree[i] = sqrt((np.linalg.norm(Theta_moy-Theta_KKT)**2 )/n) * (180/pi)   




df = pd.DataFrame({'Mean_traj (holonomique model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : BL1_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : BL1_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : BL1_RMSE_ang_degree})

df




df.to_csv('RMSE_poids_variable_BL1_MH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_BL1_MH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_BL1_MH_T=5.png')




PDFO_RMSE_PLAN = np.zeros(23)
PDFO_RMSE_ang_rad = np.zeros(23)
PDFO_RMSE_ang_degree = np.zeros(23)


for i in range (23):
    T0 = np.loadtxt(data2[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = atan(T0[3]/T0[2])

    M00 = vertcat(0,Theta_moy[1:])

    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    V1_moy = cos(Theta_moy)*T0[2]+T0[3]*sin(Theta_moy)

    V2_moy = -T0[2]*sin(Theta_moy) + cos(Theta_moy)*T0[3]

    W_moy  = (M00-T0[5])/taux

    M1 = vertcat(V1_moy[1:],0)
    M2 = vertcat(W_moy[1:],0)
    M3 = vertcat(V2_moy[1:],0)

    U1_moy = (M1-V1_moy)/taux
    U2_moy = (M2-W_moy)/taux
    U3_moy = (M3-V2_moy)/taux
    
    res = pdfo( MH_PDFO, [0.1 ,0, 0,0,0,0,0.2,0,0,0,0,0,0.7,0,0,0,0,0], constraints=Lin_const, options=options)
    a0,a1,a2,a3,a4,a5 , b0,b1,b2,b3,b4,b5, c0,c1,c2,c3,c4,c5 = res.x

    
    c1_PDFO = a0 + a1*T + a2* (T**2) + a3* (T**3) + a4* (T**4) + a5* (T**5) 
    c2_PDFO = b0 + b1*T + b2* (T**2) + b3* (T**3) + b4* (T**4) + b5* (T**5) 
    c3_PDFO = c0 + c1*T + c2* (T**2) + c3* (T**3) + c4* (T**4) + c5* (T**5)
    
    X_PDFO , Y_PDFO, Theta_PDFO = DOC_MH (c1_PDFO,c2_PDFO,c3_PDFO)
    
    PDFO_RMSE_PLAN[i] = sqrt((np.linalg.norm(Xmoy-X_PDFO)**2 + np.linalg.norm(Ymoy-Y_PDFO)**2 )/n)
    PDFO_RMSE_ang_rad[i] = sqrt((np.linalg.norm(Theta_moy-Theta_PDFO)**2 )/n)
    PDFO_RMSE_ang_degree[i] = sqrt((np.linalg.norm(Theta_moy-Theta_PDFO)**2 )/n) * (180/pi)




df = pd.DataFrame({'Mean_traj (holonomique model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : PDFO_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : PDFO_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : PDFO_RMSE_ang_degree})

df




df.to_csv('RMSE_poids_variable_PDFO_MH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_PDFO_MH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_PDFO_MH_T=5.png')

