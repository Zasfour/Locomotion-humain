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
T = np.linspace(0,5,n)




def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 



def MNH_DOC (c1,c2):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)

    opti.minimize(   (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2))   )    # ma fonction objetion

    # mes fonctions de contrainte d'égalité:

    ## pour les condition initial
    opti.subject_to( x1[0]==X1moy[0] + 10**(-4))        
    opti.subject_to( x2[0]==X2moy[0] + 10**(-4))
    opti.subject_to( x3[0]==X3moy[0] + 10**(-4))

    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )

    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)


    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux )
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux )
    opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)

    ## pour les conditions finales
    opti.subject_to( x1[-1]==X1moy[-1] + 10**(-4))
    opti.subject_to( x2[-1]==X2moy[-1] + 10**(-4))
    opti.subject_to( x3[-1]==X3moy[-1] + 10**(-4))

    #p_opts = dict(print_time = False, verbose = False)
    #s_opts = dict(print_level = 0)


    opti.solver('ipopt'  ) #, p_opts,s_opts)      


    sol = opti.solve()
    
    return sol.value(x1),sol.value(x2),sol.value(x3)




############################################# KKT

x1i = SX.sym('x1i',1)                   
x2i = SX.sym('x2i',1)                
x3i = SX.sym('x3i',1)

x1f = SX.sym('x1f',1)
x2f = SX.sym('x2f',1)
x3f = SX.sym('x3f',1)




# Je defini les vecteurs suivant :

A = SX.sym('A',6)
B = SX.sym('B',6)


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
Mue = SX.sym('Mue',1)




c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 




p1=vertcat(x1i + 10**(-4),x1_prime[2:] + 10**(-4),x1f + 10**(-4))  
g = Function('g',[x1,x1i,x1f],[p1])




Y1 = (x1_prime+taux*u1_prime*cos(x3_prime) - g(x1,x1i,x1f))
Y2 = (x2_prime+taux*u1_prime*sin(x3_prime) - g(x2,x2i,x2f)) 
Y3 = (x3_prime+taux*u2_prime - g(x3,x3i,x3f))
Y = SX.sym('Y',n+1 , 3)        ## notre contrainte

for i in range (n+1):
    Y[i,0]= Y1[i]
    Y[i,1]= Y2[i]
    Y[i,2]= Y3[i]       

for i in range (n+1):
    Y[i,0]= Y1[i]
    Y[i,1]= Y2[i]
    Y[i,2]= Y3[i]       
    
Y_function = Function('Y_function', [u1,u2,x1,x2,x3], [Y])




## notre terme qui est relié a la contrainte.
G_lambda = 0

for i in range (n+1):
    G_lambda += dot(Y[i,:], Lambda[i,:])
    
G_lambda += (u1[0]-0.0001)*Lambda[n+1,0] + (u2[0]-0.0001)*Lambda[n+1,1] + (u1[-1]-0.0001)*Lambda[n+1,2] + (u2[-1]-0.0001)*Mue


G = Function('G', [x1,x2,x3, Lambda], [G_lambda])

## notre fonction F 
F_val = (taux/2)*(dot(c1*u1,u1)+dot(c2*u2,u2))


## le Lagrangien 
L_val = F_val + G_lambda
#print(L_val.shape)




L_x = SX.zeros(5, n)

for i in range (n):
    L_x[2,i]= jacobian(L_val, x1[i])
    L_x[3,i]= jacobian(L_val, x2[i])
    L_x[4,i]= jacobian(L_val, x3[i])
#print(L_x)
    
L_u = SX.zeros(5, n)
for i in range (n):
    L_u[0,i]= jacobian(L_val, u1[i])
    L_u[1,i]= jacobian(L_val, u2[i])




R = Function ('R', [u1,u2,x1,x2,x3, Lambda, Mue,  A, B, x1i,x2i,x3i, x1f,x2f,x3f ], [(dot(L_x,L_x) + dot(L_u,L_u))])




def MNH_KKT (U1,U2,X1,X2,X3,  x1i,x2i,x3i, x1f,x2f,x3f) :

    
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    A = opti.variable(6)
    B = opti.variable(6)
    
    c1 = A[0] + A[1]*T + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0] + B[1]*T + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 


    Lambda = opti.variable(n+2,3)
    Mue = opti.variable(1)

    opti.minimize( R(U1,U2,X1,X2,X3, Lambda,Mue, A, B , x1i,x2i,x3i, x1f,x2f,x3f )) 
    
    
    for j in range (n) : 
        opti.subject_to( 0 <= c1[j])
        opti.subject_to( 0 <= c2[j] )
        opti.subject_to(  c1[j] + c2[j] == 1)

    opti.solver('ipopt')    

    sol = opti.solve()
    
    return sol.value(A), sol.value(B)




############################################# Bi-level en un coup

X1=SX.sym('X1',n)
X2=SX.sym('X2',n)  
X3=SX.sym('X3',n)  
m = SX.sym('m',1)
m = (dot(X1-x1,X1-x1) + dot(X2-x2,X2-x2) + dot(X3-x3,X3-x3))

M = Function ('M', [x1,x2,x3, X1,X2,X3], [m])



def MNH_BL1 (U1,U2,X1,X2,X3, Xi, Xf):

    opti = casadi.Opti()   

    A = opti.variable(6)
    B = opti.variable(6)
    Lambda = opti.variable(n+2,3)
    Mue = opti.variable(1)
    u1 = opti.variable(n)
    u2 = opti.variable(n)
    x1 = opti.variable(n)
    x2 = opti.variable(n)
    x3 = opti.variable(n)
    c1 = A[0]* (T**0) + A[1]* (T**1) + A[2]* (T**2) + A[3]* (T**3) + A[4]* (T**4) + A[5]* (T**5) 
    c2 = B[0]* (T**0) + B[1]* (T**1) + B[2]* (T**2) + B[3]* (T**3) + B[4]* (T**4) + B[5]* (T**5) 


    opti.minimize(5*(10**2)*R(u1,u2,x1,x2,x3, Lambda, Mue, A, B , X1[0],X2[0],X3[0], X1[-1],X2[-1],X3[-1] ) + (M(x1,x2,x3, X1,X2,X3)) )  

    for j in range (n) : 
        opti.subject_to( 0 <= c1[j])
        opti.subject_to( 0 <= c2[j] )
        opti.subject_to(  c1[j] + c2[j] == 1)

    opti.subject_to( x1[0]==Xi[0] + 10**(-4))        
    opti.subject_to( x2[0]==Xi[1] + 10**(-4))
    opti.subject_to( x3[0]==Xi[2] + 10**(-4))

    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)
    
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux )
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux )
    opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)

    opti.subject_to( x1[-1]==Xf[0] + 10**(-4))
    opti.subject_to( x2[-1]==Xf[1] + 10**(-4))
    opti.subject_to( x3[-1]==Xf[2] + 10**(-4))
    
    opti.set_initial(u1, U1)
    opti.set_initial(u2, U2)
    opti.set_initial(x1, X1)
    opti.set_initial(x2, X2)
    opti.set_initial(x3, X3)
    

    opti.solver('ipopt')      


    sol = opti.solve()
    
    return sol.value(A), sol.value(B), sol.value(x1), sol.value(x2), sol.value(x3)




######################################## Bi-level PDFO
options = {'maxfev': 10000 , 'rhobeg' : 0.001 , 'rhoend' : 1e-8}

Lin_const = []
for i in range(n):
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 1, 1))
    Lin_const.append(LinearConstraint([0, 0,0,0,0,0,1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i]], 0, 1))    
    Lin_const.append(LinearConstraint([1, T[i], (T**2)[i],(T**3)[i],(T**4)[i],(T**5)[i],0,0,0,0,0,0], 0, 1))    




def MNH_PDFO (C) :
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
    opti.subject_to( x1[0]==X1moy[0] + 10**(-4))        
    opti.subject_to( x2[0]==X2moy[0] + 10**(-4))
    opti.subject_to( x3[0]==X3moy[0] + 10**(-4))
    
    opti.subject_to( u1[0] == 0.0001 )
    opti.subject_to( u2[0] == 0.0001 )
    opti.subject_to( u1[-1] == 0.0001)
    opti.subject_to( u2[-1] == 0.0001)
    

    ## pour les contraintes d'égaliter
    opti.subject_to( u1[:n-1]*cos(x3[:n-1])==(x1[1:] + 10**(-4) - x1[:n-1])/taux)
    opti.subject_to( u1[:n-1]*sin(x3[:n-1])==(x2[1:] + 10**(-4) - x2[:n-1])/taux)
    opti.subject_to( u2[:n-1] ==(x3[1:] + 10**(-4) - x3[:n-1])/taux)
    
    ## pour les conditions finales
    opti.subject_to( x1[-1]==X1moy[-1] + 10**(-4))
    opti.subject_to( x2[-1]==X2moy[-1] + 10**(-4))
    opti.subject_to( x3[-1]==X3moy[-1] + 10**(-4))
    
    opti.solver('ipopt', {"expand":True}, {"max_iter":10000})
    
    sol = opti.solve() 
    
    X1_1 = opti.debug.value(x1)
    X2_1 = opti.debug.value(x2)
    X3_1 = opti.debug.value(x3)
    
    plt.plot(X1_1,X2_1, color = 'green')
    
    m01 = sqrt((np.linalg.norm(X1moy-X1_1)**2 + np.linalg.norm(X2moy-X2_1)**2 + np.linalg.norm(X3moy-X3_1)**2 )/n)
    
    m02 = 10*abs(sum(c1 + c2) - n)
    
    m03 = 10* mk
    
    m1 = m01+m02+m03
    return m1




KKT_RMSE_PLAN = np.zeros(23)
KKT_RMSE_ang_rad = np.zeros(23)
KKT_RMSE_ang_degree = np.zeros(23)



for i in range (23):
    T0 = np.loadtxt(data2[i])
    X1moy = T0[0]
    X2moy = T0[1]
    X3moy = atan(T0[3]/T0[2])
    M00 = vertcat(0,X3moy[1:])
    U1_moy = cos(X3moy)*T0[2]+T0[3]*sin(X3moy)
    U2_moy  = (M00-T0[5])/taux

    
    A_KKT,B_KKT = MNH_KKT (U1_moy,U2_moy,X1moy,X2moy,X3moy,  X1moy[0], X2moy[0], X3moy[0], X1moy[-1], X2moy[-1], X3moy[-1]) 
    
    c1_KKT = A_KKT[0] + A_KKT[1]*T + A_KKT[2]* (T**2) + A_KKT[3]* (T**3) + A_KKT[4]* (T**4) + A_KKT[5]* (T**5) 
    c2_KKT = B_KKT[0] + B_KKT[1]*T + B_KKT[2]* (T**2) + B_KKT[3]* (T**3) + B_KKT[4]* (T**4) + B_KKT[5]* (T**5) 
    
    X_KKT , Y_KKT, Theta_KKT = MNH_DOC (c1_KKT,c2_KKT)
    
    KKT_RMSE_PLAN[i] = sqrt((np.linalg.norm(X1moy-X_KKT)**2 + np.linalg.norm(X2moy-Y_KKT)**2 )/n)
    KKT_RMSE_ang_rad[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n)
    KKT_RMSE_ang_degree[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n) * (180/pi)   




df = pd.DataFrame({'Mean_traj (non_holonomique model KKT)' : data2, 'RMSE_plan_unity [m]' : KKT_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : KKT_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : KKT_RMSE_ang_degree})

df




df.to_csv('RMSE_poids_variable_KKT_MNH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_KKT_MNH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_KKT_MNH_T=5.png')




BL1_RMSE_PLAN = np.zeros(23)
BL1_RMSE_ang_rad = np.zeros(23)
BL1_RMSE_ang_degree = np.zeros(23)



for i in range (23):
    T0 = np.loadtxt(data2[i])
    X1moy = T0[0]
    X2moy = T0[1]
    X3moy = atan(T0[3]/T0[2])
    M00 = vertcat(0,X3moy[1:])
    U1_moy = cos(X3moy)*T0[2]+T0[3]*sin(X3moy)
    U2_moy  = (M00-T0[5])/taux
    
    Xi = [X1moy[0], X2moy[0], X3moy[0]]
    Xf = [X1moy[-1], X2moy[-1], X3moy[-1]]
    
    
    A_KKT,B_KKT,x,y,o =  MNH_BL1 (U1_moy,U2_moy,X1moy,X2moy,X3moy, Xi, Xf)
    
    c1_KKT = A_KKT[0] + A_KKT[1]*T + A_KKT[2]* (T**2) + A_KKT[3]* (T**3) + A_KKT[4]* (T**4) + A_KKT[5]* (T**5) 
    c2_KKT = B_KKT[0] + B_KKT[1]*T + B_KKT[2]* (T**2) + B_KKT[3]* (T**3) + B_KKT[4]* (T**4) + B_KKT[5]* (T**5) 
    
    X_KKT , Y_KKT, Theta_KKT = MNH_DOC (c1_KKT,c2_KKT)
    
    BL1_RMSE_PLAN[i] = sqrt((np.linalg.norm(X1moy-X_KKT)**2 + np.linalg.norm(X2moy-Y_KKT)**2 )/n)
    BL1_RMSE_ang_rad[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n)
    BL1_RMSE_ang_degree[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n) * (180/pi)   




df = pd.DataFrame({'Mean_traj (non_holonomique model Bi-level in one shot)' : data2, 'RMSE_plan_unity [m]' : BL1_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : BL1_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : BL1_RMSE_ang_degree})

df




df.to_csv('RMSE_poids_variable_BL1_MNH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_BL1_MNH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_BL1_MNH_T=5.png')




PDFO_RMSE_PLAN = np.zeros(23)
PDFO_RMSE_ang_rad = np.zeros(23)
PDFO_RMSE_ang_degree = np.zeros(23)



for i in range (23):
    T0 = np.loadtxt(data2[i])
    X1moy = T0[0]
    X2moy = T0[1]
    X3moy = atan(T0[3]/T0[2])
    M00 = vertcat(0,X3moy[1:])
    U1_moy = cos(X3moy)*T0[2]+T0[3]*sin(X3moy)
    U2_moy  = (M00-T0[5])/taux
    
    Xi = [X1moy[0], X2moy[0], X3moy[0]]
    Xf = [X1moy[-1], X2moy[-1], X3moy[-1]]
    
    
    res = pdfo( MH_PDFO, [0.8 ,0, 0,0,0,0,0.2,0,0,0,0,0], constraints=Lin_const, options=options)
    a0,a1,a2,a3,a4,a5 , b0,b1,b2,b3,b4,b5= res.x
    
    c1_KKT = a0 + a1*T + a2* (T**2) + a3* (T**3) + a4* (T**4) + a5* (T**5) 
    c2_KKT = b0 + b1*T + b2* (T**2) + b3* (T**3) + b4* (T**4) + b5* (T**5) 
    
    X_KKT , Y_KKT, Theta_KKT = MNH_DOC (c1_KKT,c2_KKT)
    
    PDFO_RMSE_PLAN[i] = sqrt((np.linalg.norm(X1moy-X_KKT)**2 + np.linalg.norm(X2moy-Y_KKT)**2 )/n)
    PDFO_RMSE_ang_rad[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n)
    PDFO_RMSE_ang_degree[i] = sqrt((np.linalg.norm(X3moy-Theta_KKT)**2 )/n) * (180/pi)   




df = pd.DataFrame({'Mean_traj (non_holonomique model Bi-level by PDFO)' : data2, 'RMSE_plan_unity [m]' : PDFO_RMSE_PLAN,
                   'RMSE_angular_unity [rad]' : PDFO_RMSE_ang_rad, 'RMSE_angular_unity [degree]' : PDFO_RMSE_ang_degree})

df



df.to_csv('RMSE_poids_variable_PDFO_MNH_T=5.csv', index = True)
KKT5 = pd.read_csv('RMSE_poids_variable_PDFO_MNH_T=5.csv')
dfi.export(KKT5, 'RMSE_poids_variable_PDFO_MNH_T=5.png')

