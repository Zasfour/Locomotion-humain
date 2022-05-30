


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


n = 500
taux = 1/n



x = SX.sym('x', n )

f= Function('f',[x],[x[1:]])

p =vertcat(x[1:],0)
g = Function ('g',[x],[p])


def Unicycle (C1) :
    C2 = 1-C1
    
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
    
    #plt.plot(X1_1,X2_1, color = 'green')
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/n)
    #m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/n)
    
    return -m1    #(-m1,-m2)


def Mombaur (alpha1, alpha2) :
    alpha3 = 1-(alpha1+alpha2)
    #print(alpha3)
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
    
    #plt.plot(X1_1,X2_1, color = 'green')
    #plt.plot(X1,X2_1, color = 'yellow')
    
    
    m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/n)
    #m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/n)
    
    return  -m1     #(-m1,-m2)


# ## Trajectoire 1

T1 = np.loadtxt("human_traj_1.dat")

X1 = T1[0]
X2 = T1[1]
X3 = T1[5]
plt.plot(X1,X2)


# In[29]:


start = time()

optimizer = BayesianOptimization(f=Unicycle, pbounds={'C1': (0, 1)}, random_state= 1 ,verbose=2)
u = optimizer.maximize(init_points= 1 , n_iter=15, xi=1e-1 ,acq='poi')
x_obs = np.array([[res['params']['C1']] for res in optimizer.res])
y_obs = np.array([res['target'] for res in optimizer.res])
i = np.where(y_obs == max(y_obs))
C1 = x_obs[i[0][0]]
F = - y_obs[i[0][0]]

plt.plot(X1,X2, color = 'r')

end = time ()



C2 = 1-C1

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
plt.plot(X1,X2, color = 'red')




m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/X1.shape[0])
m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/X1.shape[0])

print("Temps = ", end - start)
print('RMSE de la trajectoire :',m1)
print("RMSE de l'orientation :", m2)


# In[13]:


start = time()
F = np.zeros(100)
Alpha1 = np.zeros(100)
Alpha2 = np.zeros(100)

for i in range (100):
    a = uniform(0,1)
    b = uniform(0,1)
    c = uniform(0,1)    
    d = uniform(0,1) 
    plt.figure(figsize = (25,25))
    #plt.subplot(3,10,i+1)
    while not ((a<= b and c<=d and 1-(b+d)>=0 and 1-(b+d) <= 1 and 1-(a+c)<=1 and 1-(a+c)>=0  )) :
        a = uniform(0,1)
        b = uniform(0,1)
        c = uniform(0,1)    
        d = uniform(0,1)
    optimizer = BayesianOptimization(f=Mombaur, pbounds={'alpha1': (a, b), 'alpha2': (c,d)}, random_state=None,verbose=2)
    optimizer.maximize(init_points= 1 , n_iter= 5, xi= 1e-4 ,acq='poi')
    x1_obs = np.array([[res['params']['alpha1']] for res in optimizer.res])
    x2_obs = np.array([[res['params']['alpha2']] for res in optimizer.res])
    y1_obs = np.array([res['target'] for res in optimizer.res])
    j = np.where(y1_obs == max(y1_obs))
    F[i] = -y1_obs[j[0][0]]
    Alpha1[i] = x1_obs[j[0][0]]
    Alpha2[i] = x2_obs[j[0][0]]
    
    #plt.plot(X1,X2,color='r')

end = time()



i = np.where(F == min(F))
alpha1 = Alpha1[i[0][0]]
alpha2 = Alpha2[i[0][0]]
print(alpha1,alpha2)
alpha3 = 1-alpha1-alpha2

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
plt.plot(X1,X2,color='r')


m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/X1.shape[0])
m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/X1.shape[0])

print("Temps = ", end - start)
print('RMSE de la trajectoire :',m1)
print("RMSE de l'orientation :", m2)


# ## Trajectoire 2


T1 = np.loadtxt("human_traj_2.dat")

X1 = T1[0]
X2 = T1[1]
X3 = T1[5]

plt.plot(X1,X2)


# In[17]:


start = time()

optimizer = BayesianOptimization(f=Unicycle, pbounds={'C1': (0, 1)}, random_state= 1 ,verbose=2)
u = optimizer.maximize(init_points= 1 , n_iter=5, xi=1e-1 ,acq='poi')
x_obs = np.array([[res['params']['C1']] for res in optimizer.res])
y_obs = np.array([res['target'] for res in optimizer.res])
i = np.where(y_obs == max(y_obs))
C1 = x_obs[i[0][0]]
F = - y_obs[i[0][0]]

end = time ()


# In[ ]:


C2 = 1-C1

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
plt.plot(X1,X2, color = 'red')


m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/X1.shape[0])
m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/X1.shape[0])

print("Temps = ", end - start)
print('RMSE de la trajectoire :',m1)
print("RMSE de l'orientation :", m2)




start = time()
F = np.zeros(100)
Alpha1 = np.zeros(100)
Alpha2 = np.zeros(100)

for i in range (100):
    a = uniform(0,1)
    b = uniform(0,1)
    c = uniform(0,1)    
    d = uniform(0,1) 
    plt.figure(figsize = (25,25))
    #plt.subplot(3,10,i+1)
    while not ((a<= b and c<=d and 1-(b+d)>=0 and 1-(b+d) <= 1 and 1-(a+c)<=1 and 1-(a+c)>=0  )) :
        a = uniform(0,1)
        b = uniform(0,1)
        c = uniform(0,1)    
        d = uniform(0,1)
    optimizer = BayesianOptimization(f=Mombaur, pbounds={'alpha1': (a, b), 'alpha2': (c,d)}, random_state=None,verbose=2)
    optimizer.maximize(init_points= 1 , n_iter= 5, xi= 1e-4 ,acq='poi')
    x1_obs = np.array([[res['params']['alpha1']] for res in optimizer.res])
    x2_obs = np.array([[res['params']['alpha2']] for res in optimizer.res])
    y1_obs = np.array([res['target'] for res in optimizer.res])
    j = np.where(y1_obs == max(y1_obs))
    F[i] = -y1_obs[j[0][0]]
    Alpha1[i] = x1_obs[j[0][0]]
    Alpha2[i] = x2_obs[j[0][0]]
    
    #plt.plot(X1,X2,color='r')

end = time()



i = np.where(F == min(F))
alpha1 = Alpha1[i[0][0]]
alpha2 = Alpha2[i[0][0]]
print(alpha1,alpha2)
alpha3 = 1-alpha1-alpha2

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
plt.plot(X1,X2,color='r')


# In[21]:


m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/X1.shape[0])
m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/X1.shape[0])

print("Temps = ", end - start)
print('RMSE de la trajectoire :',m1)
print("RMSE de l'orientation :", m2)


# ## Trajectoire 3


T1 = np.loadtxt("human_traj_3.dat")

X1 = T1[0]
X2 = T1[1]
X3 = T1[5]

plt.plot(X1,X2)


# In[23]:


start = time()

optimizer = BayesianOptimization(f=Unicycle, pbounds={'C1': (0, 1)}, random_state= 1 ,verbose=2)
u = optimizer.maximize(init_points= 1 , n_iter=5, xi=1e-1 ,acq='poi')
x_obs = np.array([[res['params']['C1']] for res in optimizer.res])
y_obs = np.array([res['target'] for res in optimizer.res])
i = np.where(y_obs == max(y_obs))
C1 = x_obs[i[0][0]]
F = - y_obs[i[0][0]]

end = time ()


C2 = 1-C1

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
plt.plot(X1,X2, color = 'red')


start = time()
F = np.zeros(100)
Alpha1 = np.zeros(100)
Alpha2 = np.zeros(100)

for i in range (100):
    a = uniform(0,1)
    b = uniform(0,1)
    c = uniform(0,1)    
    d = uniform(0,1) 
    plt.figure(figsize = (25,25))
    #plt.subplot(3,10,i+1)
    while not ((a<= b and c<=d and 1-(b+d)>=0 and 1-(b+d) <= 1 and 1-(a+c)<=1 and 1-(a+c)>=0  )) :
        a = uniform(0,1)
        b = uniform(0,1)
        c = uniform(0,1)    
        d = uniform(0,1)
    optimizer = BayesianOptimization(f=Mombaur, pbounds={'alpha1': (a, b), 'alpha2': (c,d)}, random_state=None,verbose=2)
    optimizer.maximize(init_points= 1 , n_iter= 5, xi= 1e-4 ,acq='poi')
    x1_obs = np.array([[res['params']['alpha1']] for res in optimizer.res])
    x2_obs = np.array([[res['params']['alpha2']] for res in optimizer.res])
    y1_obs = np.array([res['target'] for res in optimizer.res])
    j = np.where(y1_obs == max(y1_obs))
    F[i] = -y1_obs[j[0][0]]
    Alpha1[i] = x1_obs[j[0][0]]
    Alpha2[i] = x2_obs[j[0][0]]
    
    #plt.plot(X1,X2,color='r')

end = time()


i = np.where(F == min(F))
alpha1 = Alpha1[i[0][0]]
alpha2 = Alpha2[i[0][0]]
print(alpha1,alpha2)
alpha3 = 1-alpha1-alpha2

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
plt.plot(X1,X2,color='r')


m1 = sqrt((np.linalg.norm(X1-X1_1)**2 + np.linalg.norm(X2-X2_1)**2 )/X1.shape[0])
m2 = sqrt((np.linalg.norm(X3-X3_1)**2 )/X1.shape[0])

print("Temps = ", end - start)
print('RMSE de la trajectoire :',m1)
print("RMSE de l'orientation :", m2)

