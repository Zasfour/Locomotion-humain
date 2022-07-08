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
        #plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Orientation")
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 



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
    opti.subject_to( v1[-1] == 0.00 ) 
    opti.subject_to( w[-1]  == 0.00 ) 
    opti.subject_to( v2[-1] == 0.00 )
    opti.subject_to( u1[-1] == 0.00 )
    opti.subject_to( u2[-1] == 0.00 )
    opti.subject_to( u3[-1] == 0.00 )
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta),sol.value(v1),sol.value(w),sol.value(v2),sol.value(u1),sol.value(u2),sol.value(u3)



xi = SX.sym('xi',1)                   
yi = SX.sym('yi',1)                
thetai = SX.sym('thetai',1)


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)

c1 = SX.sym('c1',1)
c2 = SX.sym('c2',1)
c3 = SX.sym('c3',1)
c4 = SX.sym('c4',1)


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

Lambda = SX.sym('Lambda',n+2, 6)


p1=vertcat(xi  ,x_prime[2:]  ,xf  )   
h= Function('h',[x, xi, xf],[p1])
p2=vertcat(0, v1)   
K = Function('K', [v1], [p2])
p =vertcat(v1[1:],0) 
g = Function ('g',[v1],[p])


Y1_K = (x_prime+taux*(v1_prime*cos(theta_prime) - v2_prime*sin(theta_prime)) - h(x, xi,xf))
Y2_K = (y_prime+taux*(v1_prime*sin(theta_prime) + v2_prime*cos(theta_prime)) - h(y, yi,yf)) 
Y3_K = (theta_prime+taux*w_prime - h(theta, thetai,thetaf))

Y4_K = (vertcat(0,v1[1:],v1[-1]) - vertcat(-v1[0],v1[:-1],0))/taux - vertcat(0,u1[:-1],0) 
Y5_K = (vertcat(0,w[1:],w[-1]) - vertcat(-w[0],w[:-1],0))/taux - vertcat(0,u2[:-1],0)
Y6_K = (vertcat(0,v2[1:],v2[-1]) - vertcat(-v2[0],v2[:-1],0))/taux - vertcat(0,u3[:-1],0)



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
    
G_lambda += (u1[0])*Lambda[n+1,0] + (u2[0])*Lambda[n+1,1] + (u3[0])*Lambda[n+1,2] 
G_lambda += (u1[-1])*Lambda[n+1,3] + (u2[-1])*Lambda[n+1,4] + (u3[-1])*Lambda[n+1,5] 


F_val_K =  taux*( c1 * dot(u1,u1) + c2 * dot(u2,u2) + c3 * dot(u3,u3) + c4 * Direct(y,yf,x,xf) )

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
    
    
    
R_K = Function ('R_K', [u1,u2,u3,v1,w,v2,x,y,theta, Lambda, c1, c2, c3,c4 ,xi,yi,thetai, xf,yf,thetaf  ], [dot(grad_L_K,grad_L_K)])



X1=SX.sym('X1',n)
X2=SX.sym('X2',n)  
X3=SX.sym('X3',n)  


m = SX.sym('m',1)
m = (dot(X1-x,X1-x) + dot(X2-y,X2-y) + dot(X3-theta,X3-theta))

M = Function ('M', [x,y,theta, X1,X2,X3], [m])




def MH_BL1 (U1,U2,U3 ,V1,V2,W,X,Y,THETA,Xi,Xf) :
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
    
    alpha1 = opti.variable()
    alpha2 = opti.variable()
    alpha3 = opti.variable()
    alpha4 = opti.variable()
    

    Lambda = opti.variable(n+2,6)



    opti.minimize( n * R_K(u1,u2,u3,v1,w,v2,x,y,theta, Lambda, alpha1, alpha2, alpha3, alpha4 , X[0],Y[0] ,THETA[0] , X[-1] ,Y[-1],THETA[-1] ) + M (x,y,theta, X,Y,THETA) ) 
    opti.subject_to( 0 <= alpha1)
    opti.subject_to( 0 <= alpha2 )
    opti.subject_to( 0 <= alpha3 )
    opti.subject_to( 0 <= alpha4 )
    

    opti.subject_to(  alpha1 + alpha2 + alpha3 + alpha4 == 1)
    
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
    opti.subject_to( v1[-1] == 0.00 ) 
    opti.subject_to( w[-1]  == 0.00 ) 
    opti.subject_to( v2[-1] == 0.00 )
    opti.subject_to( u1[-1] == 0.00 )
    opti.subject_to( u2[-1] == 0.00 )
    opti.subject_to( u3[-1] == 0.00 )


    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001})    
    sol = opti.solve()
    
    Alpha1 = sol.value (alpha1)
    Alpha2 = sol.value (alpha2)   
    Alpha3 = sol.value (alpha3)
    Alpha4 = sol.value (alpha4)
    
    
    return Alpha1, Alpha2, Alpha3, Alpha4 , sol.value(Lambda)



Xi = [1,0.9,pi/4]
Xf = [0,0,pi/2]

c1 = 0.2
c2 = 0.5
c3 = 0.15
c4 = 0.15

x,y,o,v1,w,v2,u1,u2,u3 = MH_DOC(c1,c2,c3,c4,Xi,Xf)



a,b,c,d , e = MH_BL1 (u1,u2,u3 ,v1,v2,w,x,y,o,Xi,Xf) 



x0,y0,o0,v01,w0,v02,u01,u02,u03 = MH_DOC(a,b,c,d,Xi,Xf)


plt.figure(figsize = (15,5))
orientation (x[0],y[0],o[0], 0.03, 0)
orientation (x[100],y[100],o[100], 0.03, 0)
orientation (x[200],y[200],o[200], 0.03, 0)
orientation (x[300],y[300],o[300], 0.03, 0)
orientation (x[400],y[400],o[400], 0.03, 1)
orientation (x[-1],y[-1],o[-1], 0.05, 0)

plt.plot(x[0],y[0],'o',label ="Point initial")
plt.plot(x[-1],y[-1],'o',label ="Point final")


plt.plot(x,y, label = 'Traj initial')
plt.plot(x0,y0, label = 'BL1')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.grid()
plt.legend()

#ax = plt.gca()
#ax.set_aspect("equal")
