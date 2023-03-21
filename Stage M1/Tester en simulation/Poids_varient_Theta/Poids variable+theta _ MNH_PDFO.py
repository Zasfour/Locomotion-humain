#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from casadi import *
from pdfo import *
import dataframe_image as dfi


n = 500
taux = 1/n
T= linspace(0,1,n)



######################################## Bi-level PDFO

options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0], 0, 1))  
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0,0,0,0,0,0,0], 0, 1))        




def Unicycle (C) :
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    c3 = C0* (T**0) + C1* (T**1) + C2* (T**2) + C3* (T**3) + C4* (T**4) + C5* (T**5)
    
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
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2)+dot(c3*(x3-X3[-1]),x3-X3[-1]))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==X1[0] + 10**(-4))        
    opti.subject_to( x2[0]==X2[0] + 10**(-4))
    opti.subject_to( x3[0]==X3[0] + 10**(-4))
    
    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)
    

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux)
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux)
    opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)
    
    ## pour les conditions finales
    opti.subject_to( x1[-1]==X1[-1] + 10**(-4))
    opti.subject_to( x2[-1]==X2[-1] + 10**(-4))
    opti.subject_to( x3[-1]==X3[-1] + 10**(-4))
    
    opti.solver('ipopt', {"expand":True}, {"max_iter":10000})
    
    sol = opti.solve() 
    
    X1_1 = opti.debug.value(x1)
    X2_1 = opti.debug.value(x2)
    X3_1 = opti.debug.value(x3)
    
    m01 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 + np.linalg.norm(X3-X3_1)**2 )/n)
    
    m02 = 10*np.abs(np.sum(c1 + c2 + c3) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return float(m1)



x1i = -4                   # condition initiale de x1
x2i = -3                 # condition initiale de x2
x3i = pi            # condition initiale de x3


x1f = 0            # condition final de x1
x2f = 0            # condition final de x2
x3f = pi/2           # condition final de x3

Xi = [x1i,x2i,x3i]
Xf = [x1f,x2f,x3f]


A = [1/16,2/16,1/16,1/16,2/16,1/16]
B = [1/20,3/20,2/20,1/20,2/20,1/20]
t = linspace (0,1,n)

c1 = A[0] + A[1]*t + A[2]* (t**2) + A[3]* (t**3) + A[4]* (t**4) + A[5]* (t**5) 
c2 = B[0] + B[1]*t + B[2]* (t**2) + B[3]* (t**3) + B[4]* (t**4) + B[5]* (t**5)
c3 = 1 - (c1 +c2)


# In[8]:


opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

u1 = opti.variable(n)
u2 = opti.variable(n)
x1 = opti.variable(n)
x2 = opti.variable(n)
x3 = opti.variable(n)

opti.minimize(   (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2)+dot(c3*(x3-x3f),x3-x3f))   )    # ma fonction objetion

# mes fonctions de contrainte d'égalité:

## pour les condition initial
opti.subject_to( x1[0]==x1i + 10**(-4))        
opti.subject_to( x2[0]==x2i + 10**(-4))
opti.subject_to( x3[0]==x3i + 10**(-4))

opti.subject_to( u1[0] == 0.0001 )
opti.subject_to( u2[0] == 0.0001 )

opti.subject_to( u1[-1] == 0.0001)
opti.subject_to( u2[-1] == 0.0001)


## pour les contraintes d'égaliter
opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux )
opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux )
opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)

## pour les conditions finales
opti.subject_to( x1[-1]==x1f + 10**(-4))
opti.subject_to( x2[-1]==x2f + 10**(-4))
opti.subject_to( x3[-1]==x3f + 10**(-4))

#p_opts = dict(print_time = False, verbose = False)
#s_opts = dict(print_level = 0)


opti.solver('ipopt'  ) #, p_opts,s_opts)      


sol = opti.solve()




U1 = sol.value(u1)
U2 = sol.value(u2)
X1 = sol.value(x1)
X2 = sol.value(x2)
X3 = sol.value(x3)



res = pdfo( Unicycle, [1/3, 0, 0, 0, 0, 0,1/3, 0,0,0,0,0,1/3, 0,0,0,0,0], constraints=Lin_const, options=options)




A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO = res.x




c1_PDFO = A0_PDFO* t**0+ A1_PDFO* t + A2_PDFO * t**2 + A3_PDFO* t**3 + A4_PDFO* t**4 + A5_PDFO* t**5
c2_PDFO = B0_PDFO* t**0+ B1_PDFO* t + B2_PDFO * t**2 + B3_PDFO* t**3 + B4_PDFO* t**4 + B5_PDFO* t**5
c3_PDFO = C0_PDFO* t**0+ C1_PDFO* t + C2_PDFO * t**2 + C3_PDFO* t**3 + C4_PDFO* t**4 + C5_PDFO* t**5




opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

u1 = opti.variable(n)
u2 = opti.variable(n)
x1 = opti.variable(n)
x2 = opti.variable(n)
x3 = opti.variable(n)

opti.minimize(   (taux/2)*(dot(C1_PDFO*u1,u1)+dot(C2_PDFO*u2,u2)+dot(C3_PDFO*(x3-x3f),x3-x3f))   )    # ma fonction objetion

# mes fonctions de contrainte d'égalité:

## pour les condition initial
opti.subject_to( x1[0]==x1i + 10**(-4))        
opti.subject_to( x2[0]==x2i + 10**(-4))
opti.subject_to( x3[0]==x3i + 10**(-4))

opti.subject_to( u1[0] == 0.0001 )
opti.subject_to( u2[0] == 0.0001 )

opti.subject_to( u1[-1] == 0.0001)
opti.subject_to( u2[-1] == 0.0001)


## pour les contraintes d'égaliter
opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux )
opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux )
opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)

## pour les conditions finales
opti.subject_to( x1[-1]==x1f + 10**(-4))
opti.subject_to( x2[-1]==x2f + 10**(-4))
opti.subject_to( x3[-1]==x3f + 10**(-4))

#p_opts = dict(print_time = False, verbose = False)
#s_opts = dict(print_level = 0)


opti.solver('ipopt'  ) #, p_opts,s_opts)      


sol = opti.solve()




X01 = sol.value(x1)
X02 = sol.value(x2)
X03 = sol.value(x3)



plt.plot(X1,X2)
plt.plot(X01,X02)




