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



def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 

def MH_DOC (c1,c2,c3,c4):

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


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *  u3 ,u3 ) + dot(c4 *  (theta-Theta_moy[-1]) ,theta-Theta_moy[-1] ) ) )


    opti.subject_to( x[0] == Xmoy[0] )       
    opti.subject_to( y[0] == Ymoy[0] )    
    opti.subject_to( theta[0] == Theta_moy[0])


    opti.subject_to( v1[0] == 0)
    opti.subject_to( w[0] ==  0)
    opti.subject_to( v2[0] == 0)
    opti.subject_to( v1[-1] == 0)
    opti.subject_to( w[-1] == 0)
    opti.subject_to( v2[-1] == 0)

    opti.subject_to( u1[-1] == 0)
    opti.subject_to( u2[-1] == 0)
    opti.subject_to( u3[-1] == 0)

    opti.subject_to( u1[0] == 0)
    opti.subject_to( u2[0] == 0)
    opti.subject_to( u3[0] == 0)

    opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )


        ## pour les conditions finales
    opti.subject_to( x[-1]==Xmoy[-1] )
    opti.subject_to( y[-1]==Ymoy[-1] )
    opti.subject_to( theta[-1]==Theta_moy[-1] )


    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001})      

    sol = opti.solve()

    return sol.value(x),sol.value(y),sol.value(theta)



xi = SX.sym('xi',1)                   
yi = SX.sym('yi',1)                
thetai = SX.sym('thetai',1)


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)


A = SX.sym('A',6)
B = SX.sym('B',6)
C = SX.sym('C',6)
D = SX.sym('D',6)


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
c4 = D[0]* (T**0) + D[1]* (T**1) + D[2]* (T**2) + D[3]* (T**3) + D[4]* (T**4) + D[5]* (T**5) 



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
 
G_lambda = 0

for i in range (n+1):
    G_lambda += dot(Y_K[i,:], Lambda[i,:])
    
G_lambda += (v1[0]-0.0001)*Lambda[n+1,0] + (w[0]-0.0001)*Lambda[n+1,1] + (v2[0]-0.0001)*Lambda[n+1,2] 
G_lambda += (v1[-1]-0.0001)*Lambda[n+1,3] + (w[-1]-0.0001)*Lambda[n+1,4] + (v2[-1]-0.0001)*Lambda[n+1,5] 

G_lambda += (u1[0]-0.0001)*Lambda[n+2,0] + (u2[0]-0.0001)*Lambda[n+2,1] + (u3[0]-0.0001)*Lambda[n+2,2] 
G_lambda += (u1[-1]-0.0001)*Lambda[n+2,3] + (u2[-1]-0.0001)*Lambda[n+2,4] + (u3[-1]-0.0001)*Lambda[n+2,5] 


F_val_K =  taux*(  dot(c1 *u1,u1) +  dot(c2 *u2,u2) +  dot(c3*u3,u3) +  dot(c4*(theta-thetaf),(theta-thetaf)))

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
    
    
    
R_K = Function ('R_K', [u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A, B, C,D ,xi,yi,thetai, xf,yf,thetaf  ], [dot(grad_L_K,grad_L_K)])
    




def MH_KKT (X,Y,Theta,V1,W,V2,U1,U2,U3):
    opti = casadi.Opti()  


    A = opti.variable(6)
    B = opti.variable(6)
    C = opti.variable(6)
    D = opti.variable(6)
    
    c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 
    c3 = C[0]* (T**0) + C[1]* (T**1) + C[2]* (T**2) + C[3]* (T**3) + C[4]* (T**4) + C[5]* (T**5) 
    c4 = D[0]* (T**0) + D[1]* (T**1) + D[2]* (T**2) + D[3]* (T**3) + D[4]* (T**4) + D[5]* (T**5) 
    

    Lambda = opti.variable(n+3,6)


    opti.minimize( R_K(U1, U2, U3,V1,W,V2,X,Y,Theta, Lambda, A,B,C,D, X[0],Y[0],Theta[0], X[-1],Y[-1],Theta[-1] )) 
    
    for j in range(n):

        opti.subject_to( 0 <= c1[j] )

        opti.subject_to( 0 <= c2[j] )

        opti.subject_to( 0 <= c3[j] )
        
        opti.subject_to( 0 <= c4[j] )
        

        opti.subject_to(  c1[j] + c2[j] + c3[j] + c4[j] == 1)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    return sol.value(A),sol.value(B),sol.value(C),sol.value(D)




X1=SX.sym('X1',n)
X2=SX.sym('X2',n)  
X3=SX.sym('X3',n)  


m = SX.sym('m',1)
m = (np.dot(X1-x,X1-x) + dot(X2-y,X2-y) + np.dot(X3-theta,X3-theta))

M = Function ('M', [x,y,theta, X1,X2,X3], [m])



def MH_BL1 (X,Y,THETA,V1,W,V2,U1,U2,U3):
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

    A = opti.variable(6)
    B = opti.variable(6)
    C = opti.variable(6)
    D = opti.variable(6)
    
    c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 
    c3 = C[0]* (T**0) + C[1]* (T**1) + C[2]* (T**2) + C[3]* (T**3) + C[4]* (T**4) + C[5]* (T**5)
    c4 = D[0]* (T**0) + D[1]* (T**1) + D[2]* (T**2) + D[3]* (T**3) + D[4]* (T**4) + D[5]* (T**5) 
    

    Lambda = opti.variable(n+3,6)


    opti.minimize( 5*10**(2) *R_K(u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A,B,C,D,  X[0],Y[0],THETA[0], X[-1],Y[-1],THETA[-1]) + M(x,y,theta, X,Y,THETA) ) 
    
    for j in range(n):

        opti.subject_to( 10**(-4) <= c1[j] )

        opti.subject_to( 10**(-4) <= c2[j] )

        opti.subject_to( 10**(-4) <= c3[j] )
        
        opti.subject_to( 10**(-4) <= c4[j] )
        

        opti.subject_to(  c1[j] + c2[j] + c3[j] + c4[j] == 1)
        
    opti.subject_to( x[0] == X[0] )        
    opti.subject_to( y[0] == Y[0] )
    opti.subject_to( theta[0] == THETA[0] )
    
    opti.subject_to( v1[0] == 0.000 )
    opti.subject_to( w[0]  == 0.000 )
    opti.subject_to( v2[0] == 0.000 )
    opti.subject_to( v1[-1] == 0.00 )
    opti.subject_to( w[-1]  == 0.00 )
    opti.subject_to( v2[-1] == 0.00 )
        
    opti.subject_to( u1[-1] == 0.00 )
    opti.subject_to( u2[-1] == 0.00 )
    opti.subject_to( u3[-1] == 0.00 )
    opti.subject_to( u1[0] == 0.000 )
    opti.subject_to( u2[0] == 0.000 )
    opti.subject_to( u3[0] == 0.000 )

    #opti.subject_to(  R_K(u1,u2,u3,v1,w,v2,x,y,theta, Lambda, A,B,C,D,  X[0],Y[0],THETA[0], X[-1],Y[-1],THETA[-1]) <= 10**(-6))
    opti.subject_to( x[1:]  == x[:n-1]+taux*(np.cos(theta[:n-1])*v1[:n-1] - np.sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(np.sin(theta[:n-1])*v1[:n-1] + np.cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:]   == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:] )  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:] ) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:] ) )


    opti.subject_to( x[-1]==X[-1] )
    opti.subject_to( y[-1]==Y[-1] )
    opti.subject_to( theta[-1]==THETA[-1] )
    
    
    opti.set_initial(u1, U1)
    opti.set_initial(u2, U2)
    opti.set_initial(u3, U3)
    opti.set_initial(v1, V1)
    opti.set_initial(w, W)
    opti.set_initial(v2, V2)
    opti.set_initial(x, X)
    opti.set_initial(y, Y)
    opti.set_initial(theta, THETA)

    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001})    

    sol = opti.solve()
    
    return sol.value(A),sol.value(B),sol.value(C),sol.value(D)



options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


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


    opti.minimize(  taux*( dot(c1 *u1,u1) +  dot(c2 *u2,u2 ) + dot(c3 *u3 ,u3 ) + dot(c4 *(theta-Theta_moy[-1]) ,theta-Theta_moy[-1] ) ) )   

        
    opti.subject_to( x[0] == Xmoy[0] )       
    opti.subject_to( y[0] == Ymoy[0] )    
    opti.subject_to( theta[0] == Theta_moy[0] )


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


    opti.subject_to( x[-1]==Xmoy[-1] )
    opti.subject_to( y[-1]==Ymoy[-1] )
    opti.subject_to( theta[-1]==Theta_moy[-1] )


    opti.solver('ipopt', {'expand' : True}, {'acceptable_constr_viol_tol':0.0001})

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    
    m01 = np.sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    
    m02 = 10*abs(np.sum(c1 + c2 + c3 + c4) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return float(m1)



BL1_rmse_plan = np.zeros(40)
BL1_rmse_ang = np.zeros(40)

PDFO_rmse_plan = np.zeros(40)
PDFO_rmse_ang = np.zeros(40)

plt.figure(figsize=(50,100))

for i in range (40):
    T0 = np.loadtxt(data[i])
    Xmoy = T0[0]
    Ymoy = T0[1]
    Theta_moy = T0[5]


    M00 = vertcat(0,Theta_moy[1:])

    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    V1_moy = np.cos(Theta_moy)*T0[2]+T0[3]*np.sin(Theta_moy)
    V2_moy = -T0[2]*np.sin(Theta_moy) + np.cos(Theta_moy)*T0[3]

    W_moy  = (M00-T0[5])/taux

    M1 = vertcat(V1_moy[1:],0)
    M2 = vertcat(W_moy[1:],0)
    M3 = vertcat(V2_moy[1:],0)

    U1_moy = (M1-V1_moy)/taux
    U2_moy = (M2-W_moy)/taux
    U3_moy = (M3-V2_moy)/taux
    
    A0,B0,C0, D0 = MH_BL1(Xmoy,Ymoy,Theta_moy,V1_moy,W_moy,V2_moy,U1_moy,U2_moy,U3_moy)
    
    C1_BL1 = A0[0] + A0[1]*T + A0[2]* (T**2) + A0[3]* (T**3) + A0[4]* (T**4) + A0[5]* (T**5)
    C2_BL1 = B0[0] + B0[1]*T + B0[2]* (T**2) + B0[3]* (T**3) + B0[4]* (T**4) + B0[5]* (T**5) 
    C3_BL1 = C0[0] + C0[1]*T + C0[2]* (T**2) + C0[3]* (T**3) + C0[4]* (T**4) + C0[5]* (T**5) 
    C4_BL1 = D0[0] + D0[1]*T + D0[2]* (T**2) + D0[3]* (T**3) + D0[4]* (T**4) + D0[5]* (T**5) 
    
    X_BL1 , Y_BL1 ,THETA_BL1 = MH_DOC(C1_BL1,C2_BL1,C3_BL1,C4_BL1)
    
    BL1_rmse_plan[i] , BL1_rmse_ang[i] = np.sqrt((np.linalg.norm(Xmoy-X_BL1)**2 + np.linalg.norm(Ymoy-Y_BL1)**2 )/n), np.sqrt(np.linalg.norm(Theta_moy-THETA_BL1)**2 /n)
    

    res = pdfo( MH_PDFO, [1/3, 0, 0, 0, 0, 0,1/3, 0,0,0,0,0,1/3, 0,0,0,0,0,0, 0,0,0,0,0], constraints=Lin_const, options=options) 
    
    A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO,D0_PDFO,D1_PDFO,D2_PDFO,D3_PDFO,D4_PDFO,D5_PDFO = res.x
    
    c1_PDFO = A0_PDFO* T**0+ A1_PDFO* T + A2_PDFO * T**2 + A3_PDFO* T**3 + A4_PDFO* T**4 + A5_PDFO* T**5
    c2_PDFO = B0_PDFO* T**0+ B1_PDFO* T + B2_PDFO * T**2 + B3_PDFO* T**3 + B4_PDFO* T**4 + B5_PDFO* T**5
    c3_PDFO = C0_PDFO* T**0+ C1_PDFO* T + C2_PDFO * T**2 + C3_PDFO* T**3 + C4_PDFO* T**4 + C5_PDFO* T**5
    c4_PDFO = D0_PDFO* T**0+ D1_PDFO* T + D2_PDFO * T**2 + D3_PDFO* T**3 + D4_PDFO* T**4 + D5_PDFO* T**5

    
    X_PDFO,Y_PDFO,THETA_PDFO  = MH_DOC (c1_PDFO,c2_PDFO,c3_PDFO,c4_PDFO)

    PDFO_rmse_plan [i] , PDFO_rmse_ang[i] = np.sqrt((np.linalg.norm(Xmoy-X_PDFO)**2 + np.linalg.norm(Ymoy-Y_PDFO)**2 )/n), sqrt( np.linalg.norm(Theta_moy-THETA_PDFO)**2 /n )


    plt.figure(figsize=(20,15))
    plt.subplot(1,2,1)

    plt.plot(Xmoy,Ymoy, 'r', label = 'Trajectoire moyenne de {}'.format(data[i]))
    plt.plot(X_BL1,Y_BL1, 'g', label = 'Bi-level en un coup')
    plt.plot(X_PDFO,Y_PDFO, 'y', label = 'Bi-level PDFO')
    plt.xlabel ('X[m]')
    plt.ylabel ('Y[m]')
    plt.legend()
    
    plt.subplot(1,2,2)
    

    plt.plot(T,THETA_BL1,'g', label = "variation du theta obtenu par le Bi-level en un coup")
    plt.plot(T,THETA_PDFO,'y', label = "variation du theta obtenu par le PDFO")
    plt.plot(T,Theta_moy,'r', label = "variation theta trajectoire moyenne de {}".format(data[i]))
    plt.axhline(Theta_moy[-1], color= 'black', label = "theta final")
    plt.legend()
    plt.xlabel ('Times[s]')
    plt.ylabel ('Theta[rad]')

    plt.savefig("Theta_Poids_varie_{}.png".format(data[i]))



df = pd.DataFrame({'Mean_traj (Bi-level in one shot)' : data, 'RMSE_plan_unity [m]' : BL1_rmse_plan,
                   'RMSE_angular_unity [rad]' : BL1_rmse_ang, 'RMSE_angular_unity [degree]' : BL1_rmse_ang*(180/np.pi)})


df.to_csv('RMSE_poids_variable_Theta_BL1.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_variable_Theta_BL1.csv'), 'BL1_MH_poids_varie_Theta.png')

df = pd.DataFrame({'Mean_traj (Bi-level PDFO)' : data, 'RMSE_plan_unity [m]' : PDFO_rmse_plan,
                   'RMSE_angular_unity [rad]' : PDFO_rmse_ang, 'RMSE_angular_unity [degree]' : PDFO_rmse_ang*(180/np.pi)})


df.to_csv('RMSE_poids_variable_Theta_PDFO.csv', index = True)

dfi.export(pd.read_csv('RMSE_poids_variable_Theta_PDFO.csv'), 'PDFO_MH_poids_varie_Theta.png')

BL1_MH_plan = np.zeros(40)
BL1_MH_ang = np.zeros(40)

for i in range(40):
    BL1_MH_plan[i] = round(BL1_rmse_plan[i], 7) 
    BL1_MH_ang[i] = round(BL1_rmse_ang[i], 7) 
    
BL1_MH_plan = 10**(7) * BL1_MH_plan
BL1_MH_ang = 10**(7) * BL1_MH_ang

PDFO_MH_plan = np.zeros(40)
PDFO_MH_ang = np.zeros(40)

for i in range(40):
    PDFO_MH_plan[i] = round(PDFO_rmse_plan[i], 7) 
    PDFO_MH_ang[i] = round(PDFO_rmse_ang[i], 7) 
    
PDFO_MH_plan = 10**(7) * PDFO_MH_plan
PDFO_MH_ang = 10**(7) * PDFO_MH_ang

M01 = int(sum(PDFO_MH_plan))
M02 = int(sum(PDFO_MH_ang))

M03 = int(sum(BL1_MH_plan))
M04 = int(sum(BL1_MH_ang))

x1 = np.zeros(M01)
x2 = np.zeros(M02)
x3 = np.zeros(M03)
x4 = np.zeros(M04)

k1 = 0
k2 = 0
k3 = 0
k4 = 0

for i in range (40):
    m1 = round(PDFO_MH_plan[i],0)
    x1[int(k1):int(m1+k1)] = (i+1)*np.ones(int(m1))
    k1 = k1 + m1
    
    m2 = round(PDFO_MH_ang[i],0)
    x2[int(k2):int(m2+k2)] = (i+1)*np.ones(int(m2))
    k2 = k2 + m2

    
    m3 = round(BL1_MH_plan[i],0)
    x3[int(k3):int(m3+k3)] = (i+1)*np.ones(int(m3))
    k3 = k3 + m3

    m4 = int(round(BL1_MH_ang[i],0))
    x4[int(k4):int(m4+k4)] = (i+1)*np.ones(int(m4))
    k4 = k4 + m4

plt.figure(figsize = (25,10))


plt.subplot(1,2,1)
bins = [x + 0.5 for x in range(0, 41)]
plt.hist([x1, x3], bins = bins, color = ['yellow', 'green'],
            edgecolor = 'red', hatch = '/', label = ['PDFO', 'BL1'],
            histtype = 'bar') # bar est le defaut
plt.title ('RMSE from plan (x,y) \n Bi-level method')
plt.ylabel('RMSE_plan [m] (10e-7)')
plt.xlabel('Mean trajectory')
plt.xticks(np.arange(1, 41, 1))
plt.legend()


plt.subplot(1,2,2)
bins = [x + 0.5 for x in range(0, 41)]
plt.hist([x2, x4], bins = bins, color = ['yellow', 'green'],
            edgecolor = 'red', hatch = '/', label = ['PDFO', 'BL1'],
            histtype = 'bar') # bar est le defaut
plt.title ('RMSE angular \n Bi-level method')
plt.ylabel('RMSE_angular [rad] (10e-7)')
plt.xlabel('Mean trajectory')
plt.xticks(np.arange(1, 41, 1))
plt.legend()

plt.savefig("PDFO_contre_BL1.png")