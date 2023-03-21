#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from casadi import *
from pdfo import *


# In[12]:


n = 500
taux = 1/n
T = np.linspace(0,1,n)


# In[3]:


def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 


# In[4]:


######################################## Bi-level PDFO

#bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1],[0, 1], [0, 1],[0, 1], [0, 1], [0, 1], [0, 1],[0, 1], [0, 1]])

options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],0,0,0,0,0,0], 0, 1))    


# In[5]:


def Unicycle (C) :
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5] = C
    c1 = A0* (T**0) + A1* (T**1) + A2* (T**2) + A3* (T**3) + A4* (T**4) + A5* (T**5) 
    c2 = B0* (T**0) + B1* (T**1) + B2* (T**2) + B3* (T**3) + B4* (T**4) + B5* (T**5)
    print(C)
    
    c01 = c1.copy()
    c02 = c2.copy()
    
    mk = 0
    
    for j in range (c1.shape[0]):
        if c1[j] < 0 :
            c01[j] = - c1[j] 
            mk = mk - c1[j] 
        if c2[j] < 0 :
            c02[j] = - c2[j]
            mk = mk - c2[j] 
            
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(dot(c01*u1,u1)+dot(c02*u2,u2))   )    # ma fonction objetion

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
    
    plt.plot(X1_1,X2_1, color = 'green')
    
    m01 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 + np.linalg.norm(X3-X3_1)**2 )/n)
    
    m02 = 10*abs(sum(c1 + c2) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return m1


# In[6]:


A = [1/10,3/10,2/10,1/10,2/10,1/10]

C1 = A[0]* (t**0) + A[1]* (t**1) + A[2]* (t**2) + A[3]* (t**3) + A[4]* (t**4) + A[5]* (t**5) 
C2 = 1 - C1


# In[7]:


x1i = -4                   # condition initiale de x1
x2i = -3.4                 # condition initiale de x2
x3i = pi/4              # condition initiale de x3


x1f = 0           # condition final de x1
x2f = 0            # condition final de x2
x3f = pi/2         # condition final de x3


# In[8]:


opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

u1 = opti.variable(n)
u2 = opti.variable(n)
x1 = opti.variable(n)
x2 = opti.variable(n)
x3 = opti.variable(n)

opti.minimize(   (taux/2)*(dot(C1*u1,u1)+dot(C2*u2,u2))   )    # ma fonction objetion

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


# In[9]:


U1 = sol.value(u1)
U2 = sol.value(u2)
X1 = sol.value(x1)
X2 = sol.value(x2)
X3 = sol.value(x3)


# In[13]:


res = pdfo( Unicycle, [1/2, 0, 0, 0, 0, 0,1/2, 0,0,0,0,0], constraints=Lin_const, options=options)


# In[14]:


A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO = res.x


# In[15]:


res.x


# In[16]:


c1_PDFO = A0_PDFO* t**0+ A1_PDFO* t + A2_PDFO * t**2 + A3_PDFO* t**3 + A4_PDFO* t**4 + A5_PDFO* t**5
c2_PDFO = B0_PDFO* t**0+ B1_PDFO* t + B2_PDFO * t**2 + B3_PDFO* t**3 + B4_PDFO* t**4 + B5_PDFO* t**5


# In[17]:


min(c2_PDFO+ c1_PDFO), max(c2_PDFO + c1_PDFO)


# In[18]:


opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

u1 = opti.variable(n)
u2 = opti.variable(n)
x1 = opti.variable(n)
x2 = opti.variable(n)
x3 = opti.variable(n)

opti.minimize(   (taux/2)*(dot(c1_PDFO*u1,u1)+dot(c2_PDFO*u2,u2))   )    # ma fonction objetion

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


# In[19]:


U1_PDFO = sol.value(u1)
U2_PDFO = sol.value(u2)
X1_PDFO = sol.value(x1)
X2_PDFO = sol.value(x2)
X3_PDFO = sol.value(x3)


# In[20]:


plt.plot(X1,X2)
plt.plot(X1_PDFO,X2_PDFO)


# In[21]:


sqrt((np.linalg.norm(X1-X1_PDFO)**2 + np.linalg.norm(X2-X2_PDFO)**2))/n , sqrt((np.linalg.norm(X3-X3_PDFO)**2))/n


# In[22]:


plt.plot(res.fhist)


# In[23]:


plt.figure(figsize = (20,10))

plt.subplot(1,2,1)
plt.plot(t,C1, label = 'alpha1 initial')
plt.plot(t,c1_PDFO, label = 'alpha1 PDFO')
plt.xlabel('Times[s]')
plt.ylabel('Alpha1')
plt.legend()

plt.subplot(1,2,2)
plt.plot(t,C2, label = 'alpha2 initial')
plt.plot(t,c2_PDFO, label = 'alpha2 PDFO')
plt.xlabel('Times[s]')
plt.ylabel('Alpha2')
plt.legend()


# In[24]:


plt.figure(figsize = (20,10))
plt.plot(X1,X2, 'r',label = 'trajectoire simulé')
plt.plot(X1_PDFO,X2_PDFO, label = 'PDFO')
plt.legend()

plt.xlabel('X[m]')
plt.ylabel('Y[m]')

