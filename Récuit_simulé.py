#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec
import random as random
get_ipython().run_line_magic('matplotlib', 'inline')
from casadi import *
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from random import *
from time import time


# In[2]:


def poids(m,r):
    c1 = np.linspace(0,r,m)
    c2 = np.linspace(0,r,m)
    C1 = []
    C2 = []
    
    for i in range (m):
        for j in range (m):
            if c1[i]+c2[j] == r :
                C1.append(c1[i])
                C2.append(c2[j])
                
    return C1,C2 


# In[3]:


def poids_K(m,r):
    c1 = np.linspace(0,r,m)
    c2 = np.linspace(0,r,m)
    c3 = np.linspace(0,r,m)
    C1 = []
    C2 = []
    C3 = []
    
    
    for i in range (m):
        for j in range (m):
            for k in range (m):
                if c1[i]+c2[j] + c3[k] == r :
                    C1.append(c1[i])
                    C2.append(c2[j])
                    C3.append(c3[k])
                    
                
    return C1,C2,C3


# In[4]:


n = 500
taux = 1/n

m = 1000
r = 1
C1,C2 = poids(m,r)

C = np.zeros((len(C1),2))

for i in range (len(C1)):
    C[i,0]= C1[i]
    C[i,1]= C2[i]


# In[5]:


n = 500
taux = 1/n

m = 100
r = 1
C1,C2,C3 = poids_K(m,r)

Alpha = np.zeros((len(C1),3))

Alpha[:,0]= C1
Alpha[:,1]= C2
Alpha[:,2]= C3  


# In[6]:


x = SX.sym('x', n )

f= Function('f',[x],[x[1:]])

p =vertcat(x[1:],0)
g = Function ('g',[x],[p])


# In[7]:


def Unicycle (c) :
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(c[0]*dot(u1,u1)+c[1]*dot(u2,u2))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==X1[0])        
    opti.subject_to( x2[0]==X2[0])
    opti.subject_to( x3[0]==X3[0])
    
    
    opti.subject_to( u1[0] >= 0.0001 )
    opti.subject_to( u1[0] <= 0.01)
    opti.subject_to( u2[0] >= 0.0001 )
    opti.subject_to( u2[0] <= 0.01)
    opti.subject_to( u1[-1] <= 0.001)
    opti.subject_to( u2[-1] <= 0.001)
    

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux)
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux)
    opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)
    
    ## pour les conditions finales
    opti.subject_to( x1[-1]==X1[-1])
    opti.subject_to( x2[-1]==X2[-1])
    opti.subject_to( x3[-1]==X3[-1])

    opti.solver('ipopt')      # suivant la méthode de KKT


    sol = opti.solve()
    
    X1_1 = sol.value(x1)
    X2_1 = sol.value(x2)
    X3_1 = sol.value(x3)
    
    plt.plot(X1_1,X2_1, color = 'green')
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/n)
    #m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/n)
    
    return m1    


# In[8]:


C1_1 = 0.05              
C2_1 = 0.95


x1_1i = -4                   # condition initiale de x1
x2_1i = 1.2                 # condition initiale de x2
x3_1i = 0              # condition initiale de x3


x1_1f = 0           # condition final de x1
x2_1f = 0.            # condition final de x2
x3_1f = pi/10        # condition final de x3


# In[9]:


opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

u1 = opti.variable(n)
u2 = opti.variable(n)
x1 = opti.variable(n)
x2 = opti.variable(n)
x3 = opti.variable(n)

opti.minimize(   (taux/2)*(C1_1*dot(u1,u1)+C2_1*dot(u2,u2))   )    # ma fonction objetion

# mes fonctions de contrainte d'égalité:

## pour les condition initial
opti.subject_to( x1[0]==x1_1i)        
opti.subject_to( x2[0]==x2_1i)
opti.subject_to( x3[0]==x3_1i)

opti.subject_to( u1[0] >= 0.001 )
opti.subject_to( u1[0] <= 0.1)
opti.subject_to( u2[0] >= 0.001 )
opti.subject_to( u2[0] <= 0.1)
opti.subject_to( u1[-1] <= 0.1)
opti.subject_to( u2[-1] <= 0.1)


## pour les contraintes d'égaliter
opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux )
opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux )
opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)

## pour les conditions finales
opti.subject_to( x1[-1]==x1_1f)
opti.subject_to( x2[-1]==x2_1f)
opti.subject_to( x3[-1]==x3_1f)


opti.solver('ipopt')      # suivant la méthode de KKT


sol = opti.solve()


# In[10]:


U1 = sol.value(u1)
U2 = sol.value(u2)

X1 = sol.value(x1)
X2 = sol.value(x2)
X3 = sol.value(x3)
plt.plot(X1,X2)


# In[11]:


def recuit_simule_U (c , mn ):
    x = c[0]
    Resultat = np.zeros((mn,3))
    for i in range (mn):
        k = np.random.randint(0,c.shape[0])
        y = c[k]
        mx = Unicycle (x)
        my = Unicycle (y)
        
        if uniform(0,1) < np.exp((( mx - my)*(i+1))) :
            x = y
            Resultat[i,0]= my
            Resultat[i,1]= y[0]
            Resultat[i,2]= y[1]
            
        else :
            x = x
            Resultat[i,0]= mx
            Resultat[i,1]= x[0]
            Resultat[i,2]= x[1]
            
            
    return Resultat
            


# In[12]:


Resultat = recuit_simule_U (C , 50 )
plt.plot(X1,X2, color = 'red')


# In[13]:


Resultat


# In[14]:


Y = Resultat[:,0]
i = np.where(Y == min(Y))
c1 = Resultat[i[0][0],1]
c2 = Resultat[i[0][0],2]
c = (c1,c2)
Unicycle (c)
plt.plot(X1,X2, color = 'red')
min(Y)


# In[15]:


print(c1,c2)


# .
# 
# .

# In[16]:


v1 = SX.sym('v1', n)

p =vertcat(v1[1:],0)
g = Function ('g',[v1],[p])


# In[17]:


def Katja_Mombaur (Alpha):
    
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


    opti.minimize(  taux*(Alpha[0] * dot(u1,u1) + Alpha[1] * dot(u2,u2 ) + Alpha[2] * dot( u3 ,u3 ) ) )   

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == X1[0])        
    opti.subject_to( y[0] == X2[0])
    opti.subject_to( theta[0] == X3[0])

    opti.subject_to( v1[0] <= 0.01 )
    opti.subject_to( w[0] <= 0.01 )
    opti.subject_to( v2[0] <= 0.01 )
    opti.subject_to( v1[0] >= 0.0001 )
    opti.subject_to( w[0] >= 0.0001 )
    opti.subject_to( v2[0] >= 0.0001 )

    opti.subject_to( v1[-1] <= 0.01 )
    opti.subject_to( w[-1] <= 0.01 )
    opti.subject_to( v2[-1] <= 0.01 )
    opti.subject_to( v1[-1] >= 0.0001 )
    opti.subject_to( w[-1] >= 0.0001 )
    opti.subject_to( v2[-1] >= 0.0001 )

    opti.subject_to( u1[0] <= 0.01 )
    opti.subject_to( u2[0] <= 0.01 )
    opti.subject_to( u3[0] <= 0.01 )
    opti.subject_to( u1[-1] <= 0.01 )
    opti.subject_to( u2[-1] <= 0.01 )
    opti.subject_to( u3[-1] <= 0.01 )


    ## pour les contraintes d'égaliter
    opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (g(v1)-v1)/taux  == u1 )
    opti.subject_to( (g(w)-w)/taux  == u2 )
    opti.subject_to( (g(v2)-v2)/taux  == u3 )


    ## pour les conditions finales
    opti.subject_to( x[-1]==X1[-1])
    opti.subject_to( y[-1]==X2[-1])
    opti.subject_to( theta[-1]==X3[-1])


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    plt.plot(X1_1,X2_1, color = 'green')
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/n)

    
    return m1    


# In[18]:


def recuit_simule_M (Alpha , mn ):
    x = Alpha[0]
    Resultat = np.zeros((mn,4))
    for i in range (mn):
        k = np.random.randint(0,Alpha.shape[0])
        y = Alpha[k]
        mx = Katja_Mombaur (x)
        my = Katja_Mombaur (y)
        
        if uniform(0,1) < np.exp((( mx - my)*(i+1))) :
            x = y
            Resultat[i,0]= my
            Resultat[i,1]= y[0]
            Resultat[i,2]= y[1]
            Resultat[i,3]= y[2]
            
            
        else :
            x = x
            Resultat[i,0]= mx
            Resultat[i,1]= x[0]
            Resultat[i,2]= x[1]
            Resultat[i,3]= x[2]
            
    return Resultat


# In[19]:


alpha1 = 0.15
alpha2 = 0.45
alpha3 = 0.4
x0 = -4
y0 = -0.9
theta0 = -pi/2

xf = 0
yf = 0
thetaf = pi/2

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


opti.minimize(  taux*(alpha1 * dot(u1,u1) + alpha2 * dot(u2,u2 ) + alpha3 * dot( u3 ,u3 ) ) )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
opti.subject_to( x[0] == x0)        
opti.subject_to( y[0] == y0)
opti.subject_to( theta[0] == theta0)

opti.subject_to( v1[0] <= 0.01 )
opti.subject_to( w[0] <= 0.01 )
opti.subject_to( v2[0] <= 0.01 )
opti.subject_to( v1[0] >= 0.0001 )
opti.subject_to( w[0] >= 0.0001 )
opti.subject_to( v2[0] >= 0.0001 )

opti.subject_to( v1[-1] <= 0.01 )
opti.subject_to( w[-1] <= 0.01 )
opti.subject_to( v2[-1] <= 0.01 )
opti.subject_to( v1[-1] >= 0.0001 )
opti.subject_to( w[-1] >= 0.0001 )
opti.subject_to( v2[-1] >= 0.0001 )

opti.subject_to( u1[0] <= 0.01 )
opti.subject_to( u2[0] <= 0.01 )
opti.subject_to( u3[0] <= 0.01 )
opti.subject_to( u1[-1] <= 0.01 )
opti.subject_to( u2[-1] <= 0.01 )
opti.subject_to( u3[-1] <= 0.01 )


## pour les contraintes d'égaliter
opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
opti.subject_to( (g(v1)-v1)/taux  == u1 )
opti.subject_to( (g(w)-w)/taux  == u2 )
opti.subject_to( (g(v2)-v2)/taux  == u3 )
    

    ## pour les conditions finales
opti.subject_to( x[-1]==xf)
opti.subject_to( y[-1]==yf)
opti.subject_to( theta[-1]==thetaf)


opti.solver('ipopt')      # suivant la méthode de KKT

sol = opti.solve()


# In[20]:


U1_1 = sol.value(u1)
U2_1 = sol.value(u2)
U3_1 = sol.value(u3)
V1_1 = sol.value(v1)
W_1 = sol.value(w)
V2_1 = sol.value(v2)
X1 = sol.value(x)
X2 = sol.value(y)
X3 = sol.value(theta)


# In[21]:


Resultat = recuit_simule_M (Alpha , 150 )
plt.plot(X1,X2, color = 'red')


# In[22]:


Resultat


# In[23]:


Y = Resultat[:,0]
i = np.where(Y == min(Y))
c1 = Resultat[i[0][0],1]
c2 = Resultat[i[0][0],2]
c3 = Resultat[i[0][0],3]

c = (c1,c2,c3)
Katja_Mombaur (c)
plt.plot(X1,X2, color = 'red')


# In[24]:


c

