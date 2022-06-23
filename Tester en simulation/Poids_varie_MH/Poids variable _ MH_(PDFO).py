#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from casadi import *
from pdfo import *


# In[2]:


n = 500
taux = 1/n
t= linspace(0,1,n)


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


options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}


Lin_const = []

for i in range(n):
    Lin_const.append(LinearConstraint([1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,0, 0,0,0,0,0,1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],0, 0,0,0,0,0], 0, 1))    
    Lin_const.append(LinearConstraint([1, t[i], (t**2)[i],(t**3)[i],(t**4)[i],(t**5)[i],0,0,0,0,0,0,0, 0,0,0,0,0], 0, 1))    


# In[5]:


def MH (C):
    [A0,A1,A2,A3,A4,A5,B0,B1,B2,B3,B4,B5,C0,C1,C2,C3,C4,C5] = C
    c1 = A0* (t**0) + A1* (t**1) + A2* (t**2) + A3* (t**3) + A4* (t**4) + A5* (t**5) 
    c2 = B0* (t**0) + B1* (t**1) + B2* (t**2) + B3* (t**3) + B4* (t**4) + B5* (t**5)
    c3 = C0* (t**0) + C1* (t**1) + C2* (t**2) + C3* (t**3) + C4* (t**4) + C5* (t**5)
    
    print(C)
    
    c01 = c1.copy()
    c02 = c2.copy()
    c03 = c3.copy()
    
    
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
    opti.subject_to( x[0] == X1[0] + 10**(-4))       
    opti.subject_to( y[0] == X2[0] + 10**(-4))    
    opti.subject_to( theta[0] == X3[0] + 10**(-4))


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
    opti.subject_to( x[-1]==X1[-1] + 10**(-4))
    opti.subject_to( y[-1]==X2[-1] + 10**(-4))
    opti.subject_to( theta[-1]==X3[-1] + 10**(-4))


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    
    m01 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 + np.linalg.norm(X3-X3_1)**2 )/n)
    
    m02 = 10*abs(sum(c1 + c2 + c3) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    
    return m1


# In[6]:


t = np.linspace(0,1,n)
A = [1/16,2/16,1/16,1/16,2/16,1/16]
B = [1/20,3/20,2/20,1/20,2/20,1/20]

Alpha1 = A[0] + A[1]*t + A[2]* (t**2) + A[3]* (t**3) + A[4]* (t**4) + A[5]* (t**5)
Alpha2 = B[0] + B[1]*t + B[2]* (t**2) + B[3]* (t**3) + B[4]* (t**4) + B[5]* (t**5) 

Alpha3 = 1 - (Alpha1+Alpha2)


# In[7]:


plt.plot(Alpha1)
plt.plot(Alpha2)
plt.plot(Alpha3)


# In[8]:


x0 = -1.5
y0 = 1.2
theta0 = pi/10

xf = 0
yf = 0
thetaf = 0


# In[9]:


x0 = -4
y0 = -0.9
theta0 = pi

xf = 0
yf = 0
thetaf = pi/2


# In[10]:


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


opti.minimize(  taux*( dot(Alpha1 *u1,u1) +  dot(Alpha2 *u2,u2 ) + dot(Alpha3 *  u3 ,u3 ) ) )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

## pour les condition initial
opti.subject_to( x[0] == x0 + 10**(-4))       
opti.subject_to( y[0] == y0 + 10**(-4))    
opti.subject_to( theta[0] == theta0 + 10**(-4))

        
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
opti.subject_to( x[-1]==xf + 10**(-4))
opti.subject_to( y[-1]==yf + 10**(-4))
opti.subject_to( theta[-1]==thetaf + 10**(-4))


opti.solver('ipopt')      # suivant la méthode de KKT

sol = opti.solve()


# In[11]:


U1 = sol.value(u1)
U2 = sol.value(u2)
U3 = sol.value(u3)
V1 = sol.value(v1)
W = sol.value(w)
V2 = sol.value(v2)
X1 = sol.value(x)
X2 = sol.value(y)
X3 = sol.value(theta)


# In[12]:


res = pdfo( MH, [1/2, 0, 0, 0, 0, 0,1/2, 0,0,0,0,0,0, 0,0,0,0,0], constraints=Lin_const, options=options)


# In[13]:


A0_PDFO,A1_PDFO,A2_PDFO,A3_PDFO,A4_PDFO,A5_PDFO,B0_PDFO,B1_PDFO,B2_PDFO,B3_PDFO,B4_PDFO,B5_PDFO,C0_PDFO,C1_PDFO,C2_PDFO,C3_PDFO,C4_PDFO,C5_PDFO = res.x


# In[14]:


c1_PDFO = A0_PDFO* t**0+ A1_PDFO* t + A2_PDFO * t**2 + A3_PDFO* t**3 + A4_PDFO* t**4 + A5_PDFO* t**5
c2_PDFO = B0_PDFO* t**0+ B1_PDFO* t + B2_PDFO * t**2 + B3_PDFO* t**3 + B4_PDFO* t**4 + B5_PDFO* t**5
c3_PDFO = C0_PDFO* t**0+ C1_PDFO* t + C2_PDFO * t**2 + C3_PDFO* t**3 + C4_PDFO* t**4 + C5_PDFO* t**5


# In[15]:


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


opti.minimize(  taux*( dot(c1_PDFO *u1,u1) +  dot(c2_PDFO *u2,u2 ) + dot(c3_PDFO *  u3 ,u3 ) ) )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

## pour les condition initial
opti.subject_to( x[0] == x0 + 10**(-4))       
opti.subject_to( y[0] == y0 + 10**(-4))    
opti.subject_to( theta[0] == theta0 + 10**(-4))

        
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
opti.subject_to( x[-1]==xf + 10**(-4))
opti.subject_to( y[-1]==yf + 10**(-4))
opti.subject_to( theta[-1]==thetaf + 10**(-4))


opti.solver('ipopt')      # suivant la méthode de KKT

sol = opti.solve()


# In[16]:


X1_PDFO = sol.value(x)
X2_PDFO = sol.value(y)
X3_PDFO = sol.value(theta)


# In[17]:


plt.plot(X1,X2)
plt.plot(X1_PDFO,X2_PDFO)

