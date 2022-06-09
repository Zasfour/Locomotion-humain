#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
get_ipython().run_line_magic('matplotlib', 'inline')
from casadi import *
from pdfo import *


# In[3]:


n = 500
T = 1
taux = T/n


# In[4]:


x1i = SX.sym('x1i',1)                   
x2i = SX.sym('x2i',1)                
x3i = SX.sym('x3i',1)


x1f = SX.sym('x1f',1)
x2f = SX.sym('x2f',1)
x3f = SX.sym('x3f',1)


c1 = SX.sym('c1',1)
c2 = SX.sym('c2',1)

u1=SX.sym('u1',n)  
u1_prime = SX.sym('u1_prime', n+1)
u1_prime[0] = 0
u1_prime[n] = 0
u1_prime[1:n] =u1[0:n-1]

u2=SX.sym('u2',n)  
u2_prime = SX.sym('u2_prime', n+1)
u2_prime[0] = 0
u2_prime[n] = 0
u2_prime[1:n] =u2[0:n-1]

x1=SX.sym('x1',n)
x1_prime = SX.sym('x1_prime', n+1)
x1_prime[0] = x1[0]
x1_prime[1:] =x1


x2=SX.sym('x2',n)
x2_prime = SX.sym('x1_prime', n+1)
x2_prime[0] = x2[0]
x2_prime[1:] =x2

x3=SX.sym('x3',n)
x3_prime = SX.sym('x1_prime', n+1)
x3_prime[0] = x3[0]
x3_prime[1:] =x3

Lambda = SX.sym('Lambda',n+2, 3)
Mue = SX.sym('Lambda',1)


# In[5]:


p1=vertcat(x1i,x1_prime[2:],x1f)   
g= Function('g',[x1, x1i, x1f],[p1])


# In[7]:


Y1_U = (x1_prime+taux*u1_prime*cos(x3_prime) - g(x1, x1i,x1f))
Y2_U = (x2_prime+taux*u1_prime*sin(x3_prime) - g(x2, x2i,x2f)) 
Y3_U = (x3_prime+taux*u2_prime - g(x3, x3i,x3f))
Y_U = SX.sym('Y',n+1 , 3)        ## notre contrainte

for i in range (n+1):
    Y_U[i,0]= Y1_U[i]
    Y_U[i,1]= Y2_U[i]
    Y_U[i,2]= Y3_U[i]    


# In[9]:


## notre terme qui est relié a la contrainte.
G_lambda = 0

for i in range (n+1):
    G_lambda += dot(Y_U[i,:], Lambda[i,:])
    
G_lambda += (u1[0]-0.0001)*Lambda[n+1,0] + (u2[0]-0.0001)*Lambda[n+1,1] + (u1[-1]-0.0001)*Lambda[n+1,2] + (u2[-1]-0.0001)*Mue


    
G_U = Function('G_U', [x1,x2,x3, Lambda], [G_lambda])


## notre fonction F 
F_val_U = (taux/2)*(c1*dot(u1,u1)+c2*dot(u2,u2))


## le Lagrangien 
L_val_U = F_val_U + G_lambda


# In[10]:


grad_L_U = SX.zeros(5, n)
for i in range (n):
    grad_L_U[0,i]= jacobian(L_val_U, u1[i])
    grad_L_U[1,i]= jacobian(L_val_U, u2[i])
    grad_L_U[2,i]= jacobian(L_val_U, x1[i])
    grad_L_U[3,i]= jacobian(L_val_U, x2[i])
    grad_L_U[4,i]= jacobian(L_val_U, x3[i])
    
    
R_U = Function ('R_U', [u1,u2,x1,x2,x3, Lambda,Mue, c1, c2, x1i,x2i,x3i, x1f,x2f,x3f ], [(dot(grad_L_U,grad_L_U))])


# In[11]:


def BL (U1,U2,X1,X2,X3, C1,C2, Xi, Xf):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    c1 = opti.variable(1)
    c2 = opti.variable(1)
    Lambda = opti.variable(n+2,3)
    Mue = opti.variable(1)
    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(R_U(u1,u2,x1,x2,x3, Lambda, Mue, c1, c2 , X1[0],X2[0],X3[0], X1[-1],X2[-1],X3[-1] ))  

    # mes fonctions de contrainte d'égalité:
    opti.subject_to( 0 <= c1)
    opti.subject_to( 0 <= c2 )
    opti.subject_to(  c1 + c2 == 1)
    
    ## pour les condition initial
    opti.subject_to( x1[0]==Xi[0])        
    opti.subject_to( x2[0]==Xi[1])
    opti.subject_to( x3[0]==Xi[2])

    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux )
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux )
    opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)


    ## pour les conditions finales
    opti.subject_to( x1[-1]==Xf[0])
    opti.subject_to( x2[-1]==Xf[1])
    opti.subject_to( x3[-1]==Xf[2])
    
    opti.set_initial(c1, C1)
    opti.set_initial(c2, C2)
    
    opti.set_initial(u1, U1)
    opti.set_initial(u2, U2)
    opti.set_initial(x1, X1)
    opti.set_initial(x2, X2)
    opti.set_initial(x3, X3)
    

    opti.solver('ipopt')      # suivant la méthode de KKT


    sol = opti.solve()
    
    return sol.value(c1), sol.value(c2), sol.value(u1), sol.value(u2), sol.value(x1), sol.value(x2), sol.value(x3)


# In[ ]:




