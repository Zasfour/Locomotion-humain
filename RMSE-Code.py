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
import dataframe_image as dfi


# In[ ]:


# Modèle Unicycle


# In[ ]:


### DOC


# In[5]:


n = 500
T = 1
taux = T/n
taux1 = 1/n


# In[6]:


def Unicycle_DOC ( Xi , Xf , c1 , c2) :
    x1i = Xi[0] 
    x2i = Xi[1]
    x3i = Xi[2]
    
    x1f = Xf[0] 
    x2f = Xf[1]
    x3f = Xf[2]
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux1/2)*(c1*dot(u1,u1)+c2*dot(u2,u2))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==x1i)        
    opti.subject_to( x2[0]==x2i)
    opti.subject_to( x3[0]==x3i)
    
    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )

    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)

    ## pour les contraintes d'égaliter
    opti.subject_to( x1[:n-1]+taux1*u1[:n-1]*cos(x3[:n-1])==x1[1:] )
    opti.subject_to( x2[:n-1]+taux1*u1[:n-1]*sin(x3[:n-1])==x2[1:] )
    opti.subject_to( x3[:n-1]+taux1*u2[:n-1] ==x3[1:])

    ## pour les conditions finales
    opti.subject_to( x1[-1]==x1f)
    opti.subject_to( x2[-1]==x2f)
    opti.subject_to( x3[-1]==x3f)


    opti.solver('ipopt')      # suivant la méthode de KKT


    sol = opti.solve()
    
    X1 = sol.value(x1)
    X2 = sol.value(x2)
    X3 = sol.value(x3)
    
    U1 = sol.value(u1)
    U2 = sol.value(u2)
    
    return X1,X2,X3,U1,U2


# In[ ]:


### IOC 


# In[7]:


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


# In[8]:


p1=vertcat(x1i,x1_prime[2:],x1f)   # Je définis un nouveau vecteur suivant x1 en prenant les n-1 dernières valeurs 
                               #  et la nième valeur vaut x1f
g= Function('g',[x1, x1i, x1f],[p1])


# In[9]:


Y1_U = (x1_prime+taux1*u1_prime*cos(x3_prime) - g(x1, x1i,x1f))
Y2_U = (x2_prime+taux1*u1_prime*sin(x3_prime) - g(x2, x2i,x2f)) 
Y3_U = (x3_prime+taux1*u2_prime - g(x3, x3i,x3f))
Y_U = SX.sym('Y',n+1 , 3)        ## notre contrainte

for i in range (n+1):
    Y_U[i,0]= Y1_U[i]
    Y_U[i,1]= Y2_U[i]
    Y_U[i,2]= Y3_U[i]       


# In[10]:


## notre terme qui est relié a la contrainte.
G_lambda = 0

for i in range (n+1):
    G_lambda += dot(Y_U[i,:], Lambda[i,:])
    
G_lambda += (u1[0]-0.0001)*Lambda[n+1,0] + (u2[0]-0.0001)*Lambda[n+1,1] + (u1[-1]-0.0001)*Lambda[n+1,2] + (u2[-1]-0.0001)*Mue


    
G_U = Function('G_U', [x1,x2,x3, Lambda], [G_lambda])


## notre fonction F 
F_val_U = (taux1/2)*(c1*dot(u1,u1)+c2*dot(u2,u2))


## le Lagrangien 
L_val_U = F_val_U + G_lambda


# In[11]:


grad_L_U = SX.zeros(5, n)
for i in range (n):
    grad_L_U[0,i]= jacobian(L_val_U, u1[i])
    grad_L_U[1,i]= jacobian(L_val_U, u2[i])
    grad_L_U[2,i]= jacobian(L_val_U, x1[i])
    grad_L_U[3,i]= jacobian(L_val_U, x2[i])
    grad_L_U[4,i]= jacobian(L_val_U, x3[i])
    
    
R_U = Function ('R_U', [u1,u2,x1,x2,x3, Lambda,Mue, c1, c2, x1i,x2i,x3i, x1f,x2f,x3f ], [(dot(grad_L_U,grad_L_U))])


# In[12]:


def Unicycle_IOC (R_U, U1,U2,X1,X2,X3,Xi,Xf) :
    x1i = Xi[0] 
    x2i = Xi[1]
    x3i = Xi[2]
    
    x1f = Xf[0] 
    x2f = Xf[1]
    x3f = Xf[2]
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    c1    = opti.variable()
    c2    = opti.variable()

    Lambda = opti.variable(n+2,3)
    Mue = opti.variable(1)
    
    
    opti.minimize( R_U(U1,U2,X1,X2,X3, Lambda, Mue, c1, c2 , x1i,x2i,x3i, x1f,x2f,x3f ))  

    opti.subject_to( 0 <= c1)
    opti.subject_to( 0 <= c2 )
    opti.subject_to(  c1 + c2 == 1)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    return sol.value(c1), sol.value(c2) , sol.value(Lambda), sol.value(Mue)


# In[ ]:


# Modèle Katja Mombaur


# In[ ]:


## DOC


# In[13]:


x = SX.sym('x', n )
p =vertcat(x[1:],0)
f1 = Function ('f1',[x],[p])


# In[14]:


def Katja_Mombaur_DOC ( Xi, Xf, alpha1, alpha2, alpha3):
    xi = Xi[0] 
    yi = Xi[1]
    thetai = Xi[2]
    
    xf = Xf[0] 
    yf = Xf[1]
    thetaf = Xf[2]
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

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
    
    
    
    opti.minimize(  taux*(alpha1 * dot(u1,u1) + alpha2 * dot(u2,u2 ) + alpha3 * dot( u3 ,u3 ) ) )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x[0] == xi)        
    opti.subject_to( y[0] == yi)
    opti.subject_to( theta[0] == thetai)
    
    
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
    opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:])  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:]) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:]) )
    

    ## pour les conditions finales
    opti.subject_to( x[-1]==xf)
    opti.subject_to( y[-1]==yf)
    opti.subject_to( theta[-1]==thetaf)


    opti.solver('ipopt')      # suivant la méthode de KKT


    sol = opti.solve()
    
    
    X = sol.value(x)
    Y = sol.value(y)
    THETA = sol.value(theta)
    
    V1 = sol.value(v1)
    V2 = sol.value(v2)
    W = sol.value(w)
    
    U1 = sol.value(u1)
    U2 = sol.value(u2)
    U3 = sol.value(u3)
    
    return X,Y,THETA, V1,V2,W, U1,U2,U3


# In[ ]:


## IOC


# In[15]:


xi = SX.sym('xi',1)                   
yi = SX.sym('yi',1)                
thetai = SX.sym('thetai',1)


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)

alpha1 = SX.sym('alpha1',1)
alpha2 = SX.sym('alpha2',1)
alpha3 = SX.sym('alpha3',1)

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


# In[16]:


p1=vertcat(xi,x_prime[2:],xf)   # Je définis un nouveau vecteur suivant x1 en prenant les n-1 dernières valeurs 
                               #  et la nième valeur vaut x1f
h= Function('h',[x, xi, xf],[p1])

p2=vertcat(0, v1)   
K = Function('K', [v1], [p2])

p =vertcat(v1[1:],0)
g = Function ('g',[v1],[p])


# In[17]:


Y1_K = (x_prime+taux*(v1_prime*cos(theta_prime) - v2_prime*sin(theta_prime)) - h(x, xi,xf))
Y2_K = (y_prime+taux*(v1_prime*sin(theta_prime) + v2_prime*cos(theta_prime)) - h(y, yi,yf)) 
Y3_K = (theta_prime+taux*w_prime - h(theta, thetai,thetaf))

U1 = (g(v1)-v1)/taux - u1
U2 = (g(w)-w)/taux  - u2
U3 = (g(v2)-v2)/taux  - u3 

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

G_lambda += (u1[0]-0.0001)*Lambda[n+2,0] + (u2[0]-0.0001)*Lambda[n+2,1] + (u2[0]-0.0001)*Lambda[n+2,2] 
G_lambda += (u1[-1]-0.0001)*Lambda[n+2,3] + (u2[-1]-0.0001)*Lambda[n+2,4] + (u2[-1]-0.0001)*Lambda[n+2,5] 


# In[18]:


F_val_K =  taux*( alpha1 * dot(u1,u1) + alpha2 * dot(u2,u2) + alpha3 * dot(u3,u3))

## le Lagrangien 
L_val_K = F_val_K + G_lambda


# In[19]:


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
    
    
    
R_K = Function ('R_K', [u1,u2,u3,v1,w,v2,x,y,theta, Lambda, alpha1, alpha2, alpha3 ,xi,yi,thetai, xf,yf,thetaf  ], [dot(grad_L_K,grad_L_K)])
    


# In[20]:


def Katja_Mombaur_IOC (R_K,U1,U2,U3 ,V1,V2,W,X,Y,THETA,Xi,Xf) :
    xi = Xi[0] 
    yi = Xi[1]
    thetai = Xi[2]
    
    xf = Xf[0] 
    yf = Xf[1]
    thetaf = Xf[2]
    
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    alpha1 = opti.variable()
    alpha2 = opti.variable()
    alpha3 = opti.variable()
    Lambda = opti.variable(n+3,6)

    
    
    opti.minimize( R_K(U1,U2,U3,V1,W,V2,X,Y,THETA, Lambda, alpha1, alpha2, alpha3, xi,yi,thetai, xf,yf,thetaf  )) 

    opti.subject_to( 0 <= alpha1)
    opti.subject_to( 0 <= alpha2 )
    opti.subject_to( 0 <= alpha3 )
    opti.subject_to(  alpha1 + alpha2 + alpha3 == 1)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    Alpha1 = sol.value (alpha1)
    Alpha2 = sol.value (alpha2)   
    Alpha3 = sol.value (alpha3)
    
    return Alpha1, Alpha2, Alpha3 , sol.value(Lambda)


# In[ ]:


# Fonction pour le PDFO


# In[21]:


bounds = np.array([[0, 1], [0, 1]])
lin_con = LinearConstraint([1, 1], 1, 1)

options = {'maxfev': 300 , 'rhobeg' : 0.01 , 'rhoend' : 1e-6}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1], 1, 1)


# In[22]:


def Unicycle (C) :
    [C1,C2] = C
    print(C)
    
    #if C1 < 0 or C2 < 0 or C1 > 1 or C2 > 1 or C1 + C2 != 1 :
     #   return 500 
    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(C1*dot(u1,u1)+C2*dot(u2,u2))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==X1[0])        
    opti.subject_to( x2[0]==X2[0])
    opti.subject_to( x3[0]==X3[0])
    
    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)
    
    for j in range (n):
        opti.subject_to( u1[j] < 10 )
        opti.subject_to( u1[j] > -10 )
        
        opti.subject_to( u2[j] < 20 )
        opti.subject_to( u2[j] > -20 )
        
    

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux)
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux)
    opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)
    
    ## pour les conditions finales
    opti.subject_to( x1[-1]==X1[-1])
    opti.subject_to( x2[-1]==X2[-1])
    opti.subject_to( x3[-1]==X3[-1])
    
    opti.solver('ipopt', {"expand":True}, {"max_iter":6000})
    
    sol = opti.solve() 
    
    X1_1 = opti.debug.value(x1)
    X2_1 = opti.debug.value(x2)
    X3_1 = opti.debug.value(x3)
    
    plt.plot(X1_1,X2_1, color = 'green')
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 + (np.linalg.norm(X3-X3_1)**2 )/n))
    
    return m1   


# In[23]:


def Mombaur (alpha) :
    (alpha1, alpha2, alpha3) = alpha
    print(alpha)
    #if not ( 0 <= alpha1 <= 1 and 0<= alpha2<=1 and 0<= alpha3 <=1 and alpha1 + alpha2 + alpha3 ==1 ) :
     #   return 3
    
    
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
    opti.subject_to( x[0] == X1[0])        
    opti.subject_to( y[0] == X2[0])
    opti.subject_to( theta[0] == THETA[0])
    
    opti.subject_to( v1[0] == 0.0001 )
    opti.subject_to( w[0]  == 0.0001 )
    opti.subject_to( v2[0] == 0.0001 )
    opti.subject_to( v1[-1] == 0.0001 )
    opti.subject_to( w[-1]  == 0.0001 )
    opti.subject_to( v2[-1] == 0.0001 )
    
    
    
    for j in range (n): 
        opti.subject_to( (u1[j]) <= 20 )
        opti.subject_to( (u1[j]) >= -20 )
        
        opti.subject_to( (u2[j]) <= 20 )
        opti.subject_to( (u2[j]) >= -20 )
        
        opti.subject_to( (u3[j]) <= 20 )
        opti.subject_to( (u3[j]) >= -20 )
        

        
    opti.subject_to( u1[-1] == 0.0001 )
    opti.subject_to( u2[-1] == 0.0001 )
    opti.subject_to( u3[-1] == 0.0001 )

    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u3[0] == 0.0001 )



    ## pour les contraintes d'égaliter
    opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
    opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:])  )
    opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:]) )
    opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:]) )


        ## pour les conditions finales
    opti.subject_to( x[-1]==X1[-1])
    opti.subject_to( y[-1]==X2[-1])
    opti.subject_to( theta[-1]==THETA[-1])


    opti.solver('ipopt')      # suivant la méthode de KKT

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    plt.plot(X1_1,X2_1, color = 'green')

    Gamma = angle_local (X3_1, gamma)
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 + np.linalg.norm(THETA-X3_1)**2 )/n)

    
    return  m1     #(-m1,-m2)


# In[24]:


def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
    


# In[25]:


def angle_local (A, C):  ### A vecteur de l'angle global obtenu 
                              ## C le vecteur local initial
    B = np.zeros(A.shape)
    n = A.shape[0]
    B[0] = C[0]
    B[-1] = C[-1]
    for i in range (1,n-1):
        B[i] = A[i+1] - A[i]
        
    return B


# In[ ]:


### RMSE MOMBAUR


# In[46]:


Traj1_kkt_K = np.zeros((11,2))
T1 = np.loadtxt("human_traj_1.dat")

for i in range (11):
    if not i ==2 :
        X = T1[0+6*i]
        Y = T1[1+6*i]
        gamma = T1[4+6*i]
        THETA = T1[5+6*i]
        M = vertcat(0,THETA[1:])

        Xi = [X[0],Y[0],THETA[0]]
        Xf = [X[-1],Y[-1],THETA[-1]]

        V1 = T1[2+6*i]*cos(THETA) + sin(THETA)*T1[3+6*i]
        V2 = -T1[2+6*i]*sin(THETA) + cos(THETA)*T1[3+6*i]
        W  = (M-T1[5+6*i])/taux

        M1 = vertcat(V1[1:],0)
        M2 = vertcat(W[1:],0)
        M3 = vertcat(V2[1:],0)

        U1_1 = (M1-V1)/taux
        U2_1 = (M2-W)/taux
        U3_1 = (M3-V2)/taux
        
        alpha1, alpha2, alpha3, Lambda = Katja_Mombaur_IOC (R_K,U1_1,U2_1,U3_1, V1,V2,W,X,Y,THETA,Xi,Xf)

        X_S1,Y_S1,THETA_S1, V1_S1,V2_S1,W_S1, U1_S1,U2_S1,U3_S1 = Katja_Mombaur_DOC (Xi, Xf, alpha1, alpha2, alpha3)
        
        Traj1_kkt_K[i,0] = sqrt((dot(X-X_S1, X-X_S1 ) + dot(Y-Y_S1, Y-Y_S1 ))/n)
        Traj1_kkt_K[i,1] = sqrt((dot(THETA-THETA_S1, THETA-THETA_S1 ) )/n)
        


# In[47]:


Traj1_kkt_K


# In[48]:


Traj2_kkt_K = np.zeros((11,2))
T1 = np.loadtxt("human_traj_2.dat")

for i in range (11):
    if not (i ==2 or i == 8):
        X = T1[0+6*i]
        Y = T1[1+6*i]
        gamma = T1[4+6*i]
        THETA = T1[5+6*i]
        M = vertcat(0,THETA[1:])

        Xi = [X[0],Y[0],THETA[0]]
        Xf = [X[-1],Y[-1],THETA[-1]]

        V1 = T1[2+6*i]*cos(THETA) + sin(THETA)*T1[3+6*i]
        V2 = -T1[2+6*i]*sin(THETA) + cos(THETA)*T1[3+6*i]
        W  = (M-T1[5+6*i])/taux

        M1 = vertcat(V1[1:],0)
        M2 = vertcat(W[1:],0)
        M3 = vertcat(V2[1:],0)

        U1_1 = (M1-V1)/taux
        U2_1 = (M2-W)/taux
        U3_1 = (M3-V2)/taux
        
        alpha1, alpha2, alpha3, Lambda = Katja_Mombaur_IOC (R_K,U1_1,U2_1,U3_1, V1,V2,W,X,Y,THETA,Xi,Xf)

        X_S1,Y_S1,THETA_S1, V1_S1,V2_S1,W_S1, U1_S1,U2_S1,U3_S1 = Katja_Mombaur_DOC (Xi, Xf, alpha1, alpha2, alpha3)
        
        Traj2_kkt_K[i,0] = sqrt((dot(X-X_S1, X-X_S1 ) + dot(Y-Y_S1, Y-Y_S1 ))/n)
        Traj2_kkt_K[i,1] = sqrt((dot(THETA-THETA_S1, THETA-THETA_S1 ) )/n)
        


# In[49]:


Traj2_kkt_K


# In[50]:


Traj3_kkt_K = np.zeros((11,2))
T1 = np.loadtxt("human_traj_3.dat")

for i in range (11):
    if not (i == 1 or i ==2 or i == 6 or i == 8):
        X = T1[0+6*i]
        Y = T1[1+6*i]
        gamma = T1[4+6*i]
        THETA = T1[5+6*i]
        M = vertcat(0,THETA[1:])

        Xi = [X[0],Y[0],THETA[0]]
        Xf = [X[-1],Y[-1],THETA[-1]]

        V1 = T1[2+6*i]*cos(THETA) + sin(THETA)*T1[3+6*i]
        V2 = -T1[2+6*i]*sin(THETA) + cos(THETA)*T1[3+6*i]
        W  = (M-T1[5+6*i])/taux

        M1 = vertcat(V1[1:],0)
        M2 = vertcat(W[1:],0)
        M3 = vertcat(V2[1:],0)

        U1_1 = (M1-V1)/taux
        U2_1 = (M2-W)/taux
        U3_1 = (M3-V2)/taux
        
        alpha1, alpha2, alpha3, Lambda = Katja_Mombaur_IOC (R_K,U1_1,U2_1,U3_1, V1,V2,W,X,Y,THETA,Xi,Xf)

        X_S1,Y_S1,THETA_S1, V1_S1,V2_S1,W_S1, U1_S1,U2_S1,U3_S1 = Katja_Mombaur_DOC (Xi, Xf, alpha1, alpha2, alpha3)
        
        Traj3_kkt_K[i,0] = sqrt((dot(X-X_S1, X-X_S1 ) + dot(Y-Y_S1, Y-Y_S1 ))/n)
        Traj3_kkt_K[i,1] = sqrt((dot(THETA-THETA_S1, THETA-THETA_S1 )  )/n)
        


# In[51]:


Traj3_kkt_K


# In[52]:


df = pd.DataFrame({'Trajectoire_1_RMSE_(X,Y)' : Traj1_kkt_K[:,0], 'Trajectoire_1_RMSE_angulaire' : Traj1_kkt_K[:,1],
                   'Trajectoire_2_RMSE_(X,Y)' : Traj2_kkt_K[:,0], 'Trajectoire_2_RMSE_angulaire' : Traj2_kkt_K[:,1], 
                   'Trajectoire_3_RMSE_(X,Y)' : Traj3_kkt_K[:,0], 'Trajectoire_3_RMSE_angulaire' : Traj3_kkt_K[:,1]}, 
                 index = ['Trajectoire moyenne', 'Sujet 1', 'Sujet 2', 'Sujet 3', 'Sujet 4', 'Sujet 5', 'Sujet 6', 'Sujet 7', 'Sujet 8', 'Sujet 9', 'Sujet 10'])


# In[53]:


df.to_csv('GfG.csv', index = True)


# In[54]:


KKT_K = pd.read_csv("GfG.csv")
dfi.export(KKT_K, 'KKT_K.png')


# In[ ]:


### RMSE PUYDUPIN-JAMIN


# In[55]:


Traj1_kkt_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_1.dat")

for i in range (11):
    if not i ==2 :
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        U1 = T1[2+6*i]*cos(X3) + sin(X3) * T1[3+6*i]
        M = vertcat(0,X3[1:])
        U2 = (M-X3)/taux1

        x1i = X1[0]
        x2i = X2[0]
        x3i = X3[0]

        x1f = X1[-1]
        x2f = X2[-1]
        x3f = X3[-1]

        Xi0 = [X1[0],X2[0],X3[0]]
        Xf0 = [X1[-1],X2[-1],X3[-1]]
        
        c1, c2, Lambda, Mue = Unicycle_IOC (R_U, U1,U2,X1,X2,X3,Xi0,Xf0)

        X1_S1 ,X2_S1,X3_S1,U1_S1,U2_S1  = Unicycle_DOC ( Xi0 , Xf0 , c1 , c2) 
        
        Traj1_kkt_M[i,0] = sqrt((dot(X1-X1_S1, X1-X1_S1 ) + dot(X2-X2_S1, X2-X2_S1 ))/n)
        Traj1_kkt_M[i,1] = sqrt((dot(X3-X3_S1, X3-X3_S1 ))/n)


# In[56]:


Traj1_kkt_M


# In[57]:


Traj2_kkt_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_2.dat")

for i in range (11):
    if not (i ==2 or i == 8):
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        U1 = T1[2+6*i]*cos(X3) + sin(X3) * T1[3+6*i]
        M = vertcat(0,X3[1:])
        U2 = (M-X3)/taux1

        x1i = X1[0]
        x2i = X2[0]
        x3i = X3[0]

        x1f = X1[-1]
        x2f = X2[-1]
        x3f = X3[-1]

        Xi0 = [X1[0],X2[0],X3[0]]
        Xf0 = [X1[-1],X2[-1],X3[-1]]
        
        c1, c2, Lambda, Mue = Unicycle_IOC (R_U, U1,U2,X1,X2,X3,Xi0,Xf0)

        X1_S1 ,X2_S1,X3_S1,U1_S1,U2_S1  = Unicycle_DOC ( Xi0 , Xf0 , c1 , c2) 
        
        Traj2_kkt_M[i,0] = sqrt((dot(X1-X1_S1, X1-X1_S1 ) + dot(X2-X2_S1, X2-X2_S1 ))/n)
        Traj2_kkt_M[i,1] = sqrt((dot(X3-X3_S1, X3-X3_S1 ))/n)


# In[58]:


Traj2_kkt_M


# In[59]:


Traj3_kkt_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_3.dat")

for i in range (11):
    if not (i == 1 or i ==2 or i == 6 or i == 8) :
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        U1 = T1[2+6*i]*cos(X3) + sin(X3) * T1[3+6*i]
        M = vertcat(0,X3[1:])
        U2 = (M-X3)/taux1

        x1i = X1[0]
        x2i = X2[0]
        x3i = X3[0]

        x1f = X1[-1]
        x2f = X2[-1]
        x3f = X3[-1]

        Xi0 = [X1[0],X2[0],X3[0]]
        Xf0 = [X1[-1],X2[-1],X3[-1]]
        
        c1, c2, Lambda, Mue = Unicycle_IOC (R_U, U1,U2,X1,X2,X3,Xi0,Xf0)

        X1_S1 ,X2_S1,X3_S1,U1_S1,U2_S1  = Unicycle_DOC ( Xi0 , Xf0 , c1 , c2) 
        
        Traj3_kkt_M[i,0] = sqrt((dot(X1-X1_S1, X1-X1_S1 ) + dot(X2-X2_S1, X2-X2_S1 ))/n)
        Traj3_kkt_M[i,1] = sqrt((dot(X3-X3_S1, X3-X3_S1 ))/n)


# In[60]:


Traj3_kkt_M 


# In[43]:


df = pd.DataFrame({'Trajectoire_1_RMSE_(X,Y)' : Traj1_kkt_M[:,0], 'Trajectoire_1_RMSE_angulaire' : Traj1_kkt_M[:,1],
                   'Trajectoire_2_RMSE_(X,Y)' : Traj2_kkt_M[:,0], 'Trajectoire_2_RMSE_angulaire' : Traj2_kkt_M[:,1], 
                   'Trajectoire_3_RMSE_(X,Y)' : Traj3_kkt_M[:,0], 'Trajectoire_3_RMSE_angulaire' : Traj3_kkt_M[:,1]}, 
                 index = ['Trajectoire moyenne', 'Sujet 1', 'Sujet 2', 'Sujet 3', 'Sujet 4', 'Sujet 5', 'Sujet 6', 'Sujet 7', 'Sujet 8', 'Sujet 9', 'Sujet 10'])


# In[44]:


gfg1_csv_data = df.to_csv('GfG1.csv', index = True)


# In[45]:


KKT_M = pd.read_csv("GfG1.csv")
dfi.export(KKT_M, 'KKT_M.png')


# In[ ]:


### RMSE BILEVEL_M


# In[77]:


Traj1_BILEVEL_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_1.dat")

for i in range (11):
    if not ( i ==2 ) :
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        
        res = pdfo(Unicycle, [0.2 ,0.8], bounds=bounds, constraints=[lin_con], options=options)
        
        C1,C2 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

        opti.minimize(   (taux/2)*(C1*dot(u1,u1)+C2*dot(u2,u2))   )    # ma fonction objetion

            # mes fonctions de contrainte d'égalité:

            ## pour les condition initial
        opti.subject_to( x1[0]==X1[0])        
        opti.subject_to( x2[0]==X2[0])
        opti.subject_to( x3[0]==X3[0])

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u1[-1] == 0.0001)
        opti.subject_to( u2[-1] == 0.0001)
        
        for j in range (n):
            opti.subject_to( u1[j] < 10 )
            opti.subject_to( u1[j] > -10 )

            opti.subject_to( u2[j] < 20 )
            opti.subject_to( u2[j] > -20 )


            ## pour les contraintes d'égaliter
        opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux)
        opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux)
        opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)

            ## pour les conditions finales
        opti.subject_to( x1[-1]==X1[-1])
        opti.subject_to( x2[-1]==X2[-1])
        opti.subject_to( x3[-1]==X3[-1])


        opti.solver('ipopt' )  


        sol = opti.solve()

        X1_1 = opti.debug.value(x1)
        X2_1 = opti.debug.value(x2)
        X3_1 = opti.debug.value(x3)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')

        Traj1_BILEVEL_M[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj1_BILEVEL_M[i,1] = sqrt((dot(X3-X3_1, X3-X3_1 ))/n)


# In[78]:


Traj1_BILEVEL_M


# In[79]:


Traj2_BILEVEL_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_2.dat")

for i in range (11):
    if not ( i ==2  or i == 8) :
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        
        res = pdfo(Unicycle, [0.5 ,0.5], bounds=bounds, constraints=[lin_con], options=options)
        
        C1,C2 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

        opti.minimize(   (taux/2)*(C1*dot(u1,u1)+C2*dot(u2,u2))   )    # ma fonction objetion

            # mes fonctions de contrainte d'égalité:

            ## pour les condition initial
        opti.subject_to( x1[0]==X1[0])        
        opti.subject_to( x2[0]==X2[0])
        opti.subject_to( x3[0]==X3[0])

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u1[-1] == 0.0001)
        opti.subject_to( u2[-1] == 0.0001)
        
        
        for j in range (n):
            opti.subject_to( u1[j] < 10 )
            opti.subject_to( u1[j] > -10 )

            opti.subject_to( u2[j] < 20 )
            opti.subject_to( u2[j] > -20 )



            ## pour les contraintes d'égaliter
        opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux)
        opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux)
        opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)

            ## pour les conditions finales
        opti.subject_to( x1[-1]==X1[-1])
        opti.subject_to( x2[-1]==X2[-1])
        opti.subject_to( x3[-1]==X3[-1])


        opti.solver('ipopt', {"expand":True} )  


        sol = opti.solve()

        X1_1 = opti.debug.value(x1)
        X2_1 = opti.debug.value(x2)
        X3_1 = opti.debug.value(x3)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')

        Traj2_BILEVEL_M[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj2_BILEVEL_M[i,1] = sqrt((dot(X3-X3_1, X3-X3_1 ))/n)


# In[80]:


Traj2_BILEVEL_M


# In[81]:


Traj3_BILEVEL_M = np.zeros((11,2))
T1 = np.loadtxt("human_traj_3.dat")

for i in range (11):
    if not ( i == 1 or i ==2 or i == 6 or i == 8 ) :
        print(i)
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        X3 = atan(T1[3+6*i]/T1[2+6*i])
        
        res = pdfo(Unicycle, x0 = [0.25 ,0.75], bounds=bounds, constraints=[lin_con], options=options)
        
        C1,C2 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

        opti.minimize(   (taux/2)*(C1*dot(u1,u1)+C2*dot(u2,u2))   )    # ma fonction objetion

            # mes fonctions de contrainte d'égalité:

            ## pour les condition initial
        opti.subject_to( x1[0]==X1[0])        
        opti.subject_to( x2[0]==X2[0])
        opti.subject_to( x3[0]==X3[0])

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u1[-1] == 0.0001)
        opti.subject_to( u2[-1] == 0.0001)
        
        for j in range (n):
            opti.subject_to( u1[j] < 10 )
            opti.subject_to( u1[j] > -10 )

            opti.subject_to( u2[j] < 20 )
            opti.subject_to( u2[j] > -20 )



            ## pour les contraintes d'égaliter
        opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] - x1[:n-1])/taux)
        opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] - x2[:n-1])/taux)
        opti.subject_to( u2[:n-1] ==(x3[1:] - x3[:n-1])/taux)

            ## pour les conditions finales
        opti.subject_to( x1[-1]==X1[-1])
        opti.subject_to( x2[-1]==X2[-1])
        opti.subject_to( x3[-1]==X3[-1])


        opti.solver('ipopt', {"expand":True}, {"max_iter":8000} )  


        sol = opti.solve()

        X1_1 = opti.debug.value(x1)
        X2_1 = opti.debug.value(x2)
        X3_1 = opti.debug.value(x3)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')

        Traj3_BILEVEL_M[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj3_BILEVEL_M[i,1] = sqrt((dot(X3-X3_1, X3-X3_1 ))/n)


# In[82]:


Traj3_BILEVEL_M


# In[83]:


df = pd.DataFrame({'Trajectoire_1_RMSE_(X,Y)' : Traj1_BILEVEL_M[:,0], 'Trajectoire_1_RMSE_angulaire' : Traj1_BILEVEL_M[:,1],
                   'Trajectoire_2_RMSE_(X,Y)' : Traj2_BILEVEL_M[:,0], 'Trajectoire_2_RMSE_angulaire' : Traj2_BILEVEL_M[:,1], 
                   'Trajectoire_3_RMSE_(X,Y)' : Traj3_BILEVEL_M[:,0], 'Trajectoire_3_RMSE_angulaire' : Traj3_BILEVEL_M[:,1]}, 
                 index = ['Trajectoire moyenne', 'Sujet 1', 'Sujet 2', 'Sujet 3', 'Sujet 4', 'Sujet 5', 'Sujet 6', 'Sujet 7', 'Sujet 8', 'Sujet 9', 'Sujet 10'])


# In[85]:


gfg1_csv_data = df.to_csv('GfG2.csv', index = True)


# In[86]:


BILEVEL_M = pd.read_csv("GfG2.csv")
dfi.export(BILEVEL_M, 'BILEVEL_M.png')


# In[ ]:


### RMSE BILEVEL_K


# In[97]:


Traj1_BILEVEL_K = np.zeros((11,2))
T1 = np.loadtxt("human_traj_1.dat")

for i in range (11):
    if not ( i ==2  ) :
        print(i)
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        THETA = T1[5+6*i]
        gamma = T1[4+6*i]
        
        res = pdfo( Mombaur, [0.1 ,0.5, 0.4], bounds=bounds1, constraints=[lin_con1], options=options)
        
        alpha1,alpha2,alpha3 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

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
        opti.subject_to( x[0] == X1[0])        
        opti.subject_to( y[0] == X2[0])
        opti.subject_to( theta[0] == THETA[0])

        opti.subject_to( v1[0] == 0.0001 )
        opti.subject_to( w[0]  == 0.0001 )
        opti.subject_to( v2[0] == 0.0001 )
        opti.subject_to( v1[-1] == 0.0001 )
        opti.subject_to( w[-1]  == 0.0001 )
        opti.subject_to( v2[-1] == 0.0001 )



        for j in range (n): 
            opti.subject_to( (u1[j]) <= 100 )
            opti.subject_to( (u1[j]) >= -100 )

            opti.subject_to( (u2[j]) <= 100 )
            opti.subject_to( (u2[j]) >= -100 )

            opti.subject_to( (u3[j]) <= 100 )
            opti.subject_to( (u3[j]) >= -100 )



        opti.subject_to( u1[-1] == 0.0001 )
        opti.subject_to( u2[-1] == 0.0001 )
        opti.subject_to( u3[-1] == 0.0001 )

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u3[0] == 0.0001 )



        ## pour les contraintes d'égaliter
        opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
        opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:])  )
        opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:]) )
        opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:]) )


            ## pour les conditions finales
        opti.subject_to( x[-1]==X1[-1])
        opti.subject_to( y[-1]==X2[-1])
        opti.subject_to( theta[-1]==THETA[-1])


        opti.solver('ipopt')      # suivant la méthode de KKT
        
        sol = opti.solve()

        X1_1 = opti.debug.value(x)
        X2_1 = opti.debug.value(y)
        THETA_1 = opti.debug.value(theta)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')
        Gamma_1 = angle_local (THETA_1, gamma)

        Traj1_BILEVEL_K[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj1_BILEVEL_K[i,1] = sqrt((dot(THETA-THETA_1, THETA-THETA_1 )  )/n) 


# In[98]:


Traj1_BILEVEL_K


# In[115]:


Traj2_BILEVEL_K = np.zeros((11,2))
T1 = np.loadtxt("human_traj_2.dat")

for i in range (11):
    if not ( i ==2 or i == 8 ) :
        print(i)
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        THETA = T1[5+6*i]
        gamma = T1[4+6*i]
        
        res = pdfo( Mombaur, [0.75 ,0., 0.25], bounds=bounds1, constraints=[lin_con1], options=options)
        
        alpha1,alpha2,alpha3 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

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
        opti.subject_to( x[0] == X1[0])        
        opti.subject_to( y[0] == X2[0])
        opti.subject_to( theta[0] == THETA[0])

        opti.subject_to( v1[0] == 0.0001 )
        opti.subject_to( w[0]  == 0.0001 )
        opti.subject_to( v2[0] == 0.0001 )
        opti.subject_to( v1[-1] == 0.0001 )
        opti.subject_to( w[-1]  == 0.0001 )
        opti.subject_to( v2[-1] == 0.0001 )



        for j in range (n): 
            opti.subject_to( (u1[j]) <= 20 )
            opti.subject_to( (u1[j]) >= -20 )

            opti.subject_to( (u2[j]) <= 20 )
            opti.subject_to( (u2[j]) >= -20 )

            opti.subject_to( (u3[j]) <= 30 )
            opti.subject_to( (u3[j]) >= -30 )



        opti.subject_to( u1[-1] == 0.0001 )
        opti.subject_to( u2[-1] == 0.0001 )
        opti.subject_to( u3[-1] == 0.0001 )

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u3[0] == 0.0001 )



        ## pour les contraintes d'égaliter
        opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
        opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:])  )
        opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:]) )
        opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:]) )


            ## pour les conditions finales
        opti.subject_to( x[-1]==X1[-1])
        opti.subject_to( y[-1]==X2[-1])
        opti.subject_to( theta[-1]==THETA[-1])


        opti.solver('ipopt')      # suivant la méthode de KKT
        
        sol = opti.solve()

        X1_1 = opti.debug.value(x)
        X2_1 = opti.debug.value(y)
        THETA_1 = opti.debug.value(theta)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')
        Gamma_1 = angle_local (THETA_1, gamma)

        Traj2_BILEVEL_K[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj2_BILEVEL_K[i,1] = sqrt((dot(THETA-THETA_1, THETA-THETA_1 )  )/n) 


# In[116]:


Traj2_BILEVEL_K


# In[ ]:


Traj3_BILEVEL_K = np.zeros((11,2))


# In[128]:


T1 = np.loadtxt("human_traj_3.dat")

for i in range (9,10):
    if not ( i == 1 or i ==2 or i == 6 or i == 8  ) :
        print(i)
        X1 = T1[0+6*i]
        X2 = T1[1+6*i]
        THETA = T1[5+6*i]
        gamma = T1[4+6*i]
        
        res = pdfo( Mombaur, [0.1 ,0.5, 0.4], bounds=bounds1, constraints=[lin_con1], options=options)
        
        alpha1,alpha2,alpha3 = res.x
        opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

        u1 = opti.variable(n)
        u2 = opti.variable(n)
        x1 = opti.variable(n)
        x2 = opti.variable(n)
        x3 = opti.variable(n)

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
        opti.subject_to( x[0] == X1[0])        
        opti.subject_to( y[0] == X2[0])
        opti.subject_to( theta[0] == THETA[0])

        opti.subject_to( v1[0] == 0.0001 )
        opti.subject_to( w[0]  == 0.0001 )
        opti.subject_to( v2[0] == 0.0001 )
        opti.subject_to( v1[-1] == 0.0001 )
        opti.subject_to( w[-1]  == 0.0001 )
        opti.subject_to( v2[-1] == 0.0001 )



        for j in range (n): 
            opti.subject_to( (u1[j]) <= 20 )
            opti.subject_to( (u1[j]) >= -20 )

            opti.subject_to( (u2[j]) <= 20 )
            opti.subject_to( (u2[j]) >= -20 )

            opti.subject_to( (u3[j]) <= 20 )
            opti.subject_to( (u3[j]) >= -20 )



        opti.subject_to( u1[-1] == 0.0001 )
        opti.subject_to( u2[-1] == 0.0001 )
        opti.subject_to( u3[-1] == 0.0001 )

        opti.subject_to( u1[0] == 0.0001 )
        opti.subject_to( u2[0] == 0.0001 )
        opti.subject_to( u3[0] == 0.0001 )



        ## pour les contraintes d'égaliter
        opti.subject_to( x[1:] == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( y[1:] == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )
        opti.subject_to( theta[1:] == theta[:n-1] + taux*w[:n-1] )
        opti.subject_to( (v1[:n-1] + taux* u1[:n-1] == v1[1:])  )
        opti.subject_to( (v2[:n-1] + taux* u3[:n-1] == v2[1:]) )
        opti.subject_to( (w[:n-1] + taux* u2[:n-1] == w[1:]) )


            ## pour les conditions finales
        opti.subject_to( x[-1]==X1[-1])
        opti.subject_to( y[-1]==X2[-1])
        opti.subject_to( theta[-1]==THETA[-1])


        opti.solver('ipopt')      # suivant la méthode de KKT
        
        sol = opti.solve()

        X1_1 = opti.debug.value(x)
        X2_1 = opti.debug.value(y)
        THETA_1 = opti.debug.value(theta)

        plt.plot(X1_1,X2_1, color = 'green')
        plt.plot(X1,X2, color = 'red')
        Gamma_1 = angle_local (THETA_1, gamma)

        Traj3_BILEVEL_K[i,0] = sqrt((dot(X1-X1_1, X1-X1_1 ) + dot(X2-X2_1, X2-X2_1 ))/n)
        Traj3_BILEVEL_K[i,1] = sqrt((dot(THETA-THETA_1, THETA-THETA_1 )  )/n) 


# In[129]:


Traj3_BILEVEL_K


# In[130]:


df = pd.DataFrame({'Trajectoire_1_RMSE_(X,Y)' : Traj1_BILEVEL_K[:,0], 'Trajectoire_1_RMSE_angulaire' : Traj1_BILEVEL_K[:,1],
                   'Trajectoire_2_RMSE_(X,Y)' : Traj2_BILEVEL_K[:,0], 'Trajectoire_2_RMSE_angulaire' : Traj2_BILEVEL_K[:,1], 
                   'Trajectoire_3_RMSE_(X,Y)' : Traj3_BILEVEL_K[:,0], 'Trajectoire_3_RMSE_angulaire' : Traj3_BILEVEL_K[:,1]}, 
                 index = ['Trajectoire moyenne', 'Sujet 1', 'Sujet 2', 'Sujet 3', 'Sujet 4', 'Sujet 5', 'Sujet 6', 'Sujet 7', 'Sujet 8', 'Sujet 9', 'Sujet 10'])


# In[131]:


df.to_csv('GfG3.csv', index = True)


# In[133]:


BILEVEL_K = pd.read_csv("GfG3.csv")
dfi.export(BILEVEL_K, 'BILEVEL_K.png')

