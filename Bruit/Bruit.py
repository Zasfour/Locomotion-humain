#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *
from scipy.optimize import approx_fprime
import cyipopt



m01 ,m02 = 62.60648207785031 ,8716.76547379362
m11 ,m12 = 0.0 ,1154653828.5970802
m21 ,m22 = 20.543893994759976 ,7861.558309796407
m31 ,m32 = 488.0082137605221 ,489.1307329027765
m41 ,m42 = 0.003127686679216384 ,0.15123248841357761
m51 ,m52 = 41.753853472012764 ,73.75947511558543
m61 ,m62 = 0.0 ,115465.38285970801



n = 50
taux = 5/n
T = np.linspace(0,5,n)


x = SX.sym('x',n)
y = SX.sym('y',n)
xf = SX.sym('xf',1)
yf = SX.sym('yf',1)


N = SX.zeros(1)
Y = ((xf - x[1:])**2 + (yf - y[1:])**2 + 10**(-5) )/((xf - x[:-1])**2 + (yf - y[:-1])**2 + 10**(-5) )
for i in range (Y.shape[0]):
    N += Y[i]
                                                
Direct = Function('Direct', [x,xf,y,yf],[N])


M = []
for i in range (n-1):
    M = vertcat(M,fmax(0,(x[i+1]-xf)**2 + (y[i+1]-yf)**2 - ((x[i]-xf)**2 + (y[i]-yf)**2)))

N0 = SX.zeros(1)
for i in range (M.shape[0]):
    N0 += M[i]
    
Max = Function ('Max',[x,y,xf,yf],[N0])


def MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf):
    
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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * ((Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41))+ c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0 )
    opti.subject_to( w[0]  == 0 )
    opti.subject_to( v2[0] == 0 )
    opti.subject_to( u1[0] == 0 )
    opti.subject_to( u2[0] == 0 )
    opti.subject_to( u3[0] == 0 )
    

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
    opti.subject_to( v1[-1] == 0 ) 
    opti.subject_to( w[-1]  == 0 ) 
    opti.subject_to( v2[-1] == 0 )
    opti.subject_to( u1[-1] == 0 )
    opti.subject_to( u2[-1] == 0 )
    opti.subject_to( u3[-1] == 0 )
    
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta),sol.value(v1),sol.value(w),sol.value(v2),sol.value(u1),sol.value(u2),sol.value(u3)


options = {'maxfev': 100000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1], [0, 1], [0, 1] , [0, 1], [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1, 1, 1, 1 ,1], 1, 1)



def PDFO (C):
    c1,c2,c3,c4,c5,c6,c7 = C
    print(C)
    mk = 0
    
    if c1 < 0:
        c1 = -c1
        mk+= c1
    if c2 < 0:
        c2 = -c2
        mk+= c2
    if c3 < 0:
        c3 = -c3
        mk+= c3
    if c4 < 0:
        c4 = -c4
        mk+= c4
    if c5 < 0:
        c5 = -c5
        mk+= c5
    if c6 < 0:
        c6 = -c6
        mk+= c6
    if c7 < 0:
        c7 = -c7
        mk+= c7
        
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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * (Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41)+ c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0 )
    opti.subject_to( w[0]  == 0 )
    opti.subject_to( v2[0] == 0 )
    opti.subject_to( u1[0] == 0 )
    opti.subject_to( u2[0] == 0 )
    opti.subject_to( u3[0] == 0 )
    

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
    opti.subject_to( v1[-1] == 0 ) 
    opti.subject_to( w[-1]  == 0 ) 
    opti.subject_to( v2[-1] == 0 )
    opti.subject_to( u1[-1] == 0 )
    opti.subject_to( u2[-1] == 0 )
    opti.subject_to( u3[-1] == 0 )
    
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.0001} )             

    sol = opti.solve()
    
    X1_1 = sol.value(x)
    X2_1 = sol.value(y)
    X3_1 = sol.value(theta)
    
    m001 = sqrt((np.linalg.norm(X0-X1_1)**2 + np.linalg.norm(Y0-X2_1)**2 + np.linalg.norm(Theta-X3_1)**2 )/n)
    m002 = 10* np.abs (c1 + c2 + c3 + c4 + c5 + c6 + c7 -1)
    m003 = 10* mk
    
    m1 = float(m001+m002+m003)

    return m1
    


xf = SX.sym('xf',1)
yf = SX.sym('yf',1)
thetaf = SX.sym('thetaf',1)

xi = SX.sym('xi',1)
yi = SX.sym('yi',1)
thetai = SX.sym('thetai',1)


## Position
x=SX.sym('x',n)

y=SX.sym('y',n)

theta=SX.sym('theta',n)

X0=SX.sym('X0',n)

Y0=SX.sym('Y',n)

Theta=SX.sym('Theta',n)

## Vitesse
v1=SX.sym('v1',n)  

v2=SX.sym('v2',n)  

w=SX.sym('w',n)

## accélération 
u1 = SX.sym('u1',n)
u2 = SX.sym('u2',n)
u3 = SX.sym('u3',n)


## Notre fonction objectif
Objectif = SX.zeros(1)
Objectif = dot(x-X0,x-X0) + dot(y-Y0,y-Y0) + dot(theta-Theta,theta-Theta)    

F_obj = Function('F_obj',[x,y,theta,X0,Y0,Theta],[Objectif])

## gradient de la fonction objectif 
X = vertcat(x,y,theta,v1,w,v2,u1,u2,u3)
H_obj , J_obj = hessian(Objectif,X) 
Gradient_obj = Function("Gradient_obj",[u1,u2,u3,v1,w,v2,x,y,theta,X0,Y0,Theta],[J_obj])



## Contrainte 
G0 = SX.sym('G0',9)
G0[0] = x[0] - xi
G0[1] = y[0] - yi
G0[2] = theta[0] - thetai
G0[3] = v1[0]
G0[4] = w[0]
G0[5] = v2[0]
G0[6] = u1[0]
G0[7] = u2[0]
G0[8] = u3[0]
G1 = SX.sym('G1',6*(n-1))
for i in range (n-1) :
    G1[6*i] = x[i] + taux * (v1[i] * cos(theta[i]) - sin(theta[i])*v2[i])- x[i+1]
    G1[6*i+1] = y[i] + taux *(v1[i] * sin(theta[i]) + cos(theta[i])*v2[i]) - y[i+1]
    G1[6*i+2] = theta[i] + taux * w[i]  - theta[i+1]
    G1[6*i+3] = v1[i] + taux * u1[i]  - v1[i+1]
    G1[6*i+4] = w[i] + taux * u2[i]  - w[i+1]
    G1[6*i+5] = v2[i] + taux * u3[i]  - v2[i+1]
G2 = SX.sym('G2',9)
G2[0] = x[-1] - xf
G2[1] = y[-1] - yf
G2[2] = theta[-1] - thetaf
G2[3] = v1[-1]
G2[4] = w[-1]
G2[5] = v2[-1]
G2[6] = u1[-1]
G2[7] = u2[-1]
G2[8] = u3[-1]
G = vertcat(G0,G1,G2)  ### première partie des contrainte (il ne reste que le déterminant)
Contraint = Function("contrainte",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[G])



F_K = SX.sym('F_K',7)
F_K[0] = ((dot(u1,u1)-m01 )/(m02-m01))
F_K[1] = ((dot(u2,u2 )-m11 )/(m12-m11))
F_K[2] = ((dot(u3 ,u3)-m21)/(m22-m21)) 
F_K[3] = ((Direct(y,yf,x,xf)-m31 )/(m32-m31))
F_K[4] = ((Max(x,y,xf,yf)-m41 )/(m42-m41))
F_K[5] = (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51))
F_K[6] = ((dot(w,w)-m61)/(m62-m61))

F_G = vertcat(F_K ,G0, G1,G2)
J = SX.zeros(X.shape[0],F_G.shape[0])

for i in range (F_G.shape[0]):
    HJ,J[:,i] = hessian(F_G[i],X)
        
Lambda = SX.sym('Lambda',6*(n+2))
c1 = SX.sym('c1',1)
c2 = SX.sym('c2',1)
c3 = SX.sym('c3',1)
c4 = SX.sym('c4',1)
c5 = SX.sym('c5',1)
c6 = SX.sym('c6',1)
c7 = SX.sym('c7',1)

C = vertcat(c1,c2,c3,c4,c5,c6,c7,Lambda)



Res = np.matmul(J,C)
Residu = Function('Residu',[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf,c1,c2,c3,c4,c5,c6,c7,Lambda],[dot(Res,Res)])

Matrice_J = Function("Matrice_J",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[J])      ## pour le calcule du déterminant 

####### Jacobien CONTRAINTE 
Jacobien_contrainte = SX.zeros(X.shape[0]*G.shape[0])
for i in range (G.shape[0]):
    H_cont , Jacobien_contrainte[i*9*n:(i+1)*9*n] = hessian(G[i],X)
                                                        ### gradient des contrainte sauf celle du déterminant 

Jac_contrainte = Function("Jac_contrainte",[u1,u2,u3,v1,w,v2,x,y,theta,xi,yi,thetai,xf,yf,thetaf],[Jacobien_contrainte])  ### ajouter après la partie du déterminant 



def KKT (X,Y,Theta,V1,W,V2,U1,U2,U3,Xi,Xf):
    opti = casadi.Opti()   # cette fonction nous permet de trouver la solution de problème

    c1 = opti.variable()
    c2 = opti.variable()
    c3 = opti.variable()
    c4 = opti.variable()
    c5 = opti.variable()
    c6 = opti.variable()
    c7 = opti.variable()
    Lambda = opti.variable(6*(n+2))



    opti.minimize( Residu(U1,U2,U3,V1,W,V2,X,Y,Theta, Xi[0],Xi[1] ,Xi[2] , Xf[0] ,Xf[1],Xf[2], c1,c2,c3,c4,c5,c6,c7,Lambda)) 
    opti.subject_to( 0 <= c1)
    opti.subject_to( 0 <= c2 )
    opti.subject_to( 0 <= c3 )
    opti.subject_to( 0 <= c4 )
    opti.subject_to( 0 <= c5 )
    opti.subject_to( 0 <= c6 )
    opti.subject_to( 0 <= c7 )
    opti.subject_to(  c1 + c2 + c3 + c4 + c5 + c6 + c7 == 1)

    opti.solver('ipopt', {"expand" : True}, {"tol" : 10**(-5)}  ) 
    #opti.solver('ipopt')             
    
 
    sol = opti.solve()
    
    return sol.value (c1),sol.value (c2),sol.value (c3),sol.value (c4),sol.value (c5),sol.value (c6),sol.value (c7)
    



class BL1(object):

    def __init__(self,X0,Y0,Theta,xi,yi,thetai,xf,yf,thetaf, taux):
        self.X0 = X0
        self.Y0 = Y0
        self.Theta = Theta
        self.xi = xi
        self.yi = yi
        self.thetai = thetai
        self.xf = xf
        self.yf = yf
        self.thetaf = thetaf
        self.taux = taux
        
        
    def objective(self, x):
        return F_obj(x[:n],x[n:2*n],x[2*n:3*n],self.X0,self.Y0,self.Theta)[0]
    
    
    def gradient(self, x):
        return np.array(Gradient_obj(x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n],self.X0,self.Y0,self.Theta))
    
    def constraint_0(self,x):
        c0 = np.array(Contraint(x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n],self.xi,self.yi,self.thetai,self.xf,self.yf,self.thetaf))
        return c0  
    
    def constraint_1(self,x):
        J0 = np.matrix(Matrice_J(x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n],self.xi,self.yi,self.thetai,self.xf,self.yf,self.thetaf))
        c1 = np.linalg.det(J0.T @J0)
        c01 = np.zeros(1)
        c01[0] = c1
        return c01
        

    def constraints(self, x):
        cte0 = self.constraint_0(x)
        cte1 = self.constraint_1(x)
        cte = np.zeros(6*(n+2)+1)
        cte[:6*(n+2)] = cte0[:,0]
        cte[-1] = cte1
        return cte
    
    def jacobian(self, x):
        jac1 = np.array(Jac_contrainte(x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n],self.xi,self.yi,self.thetai,self.xf,self.yf,self.thetaf))[:,0]
        jac2 = approx_fprime(x, self.constraint_1,self.taux)
        jac = np.concatenate((jac1, jac2), axis=0)
        return jac
    

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


T0 = np.loadtxt('S1500.dat')
X0_ = T0[0]
Y0_ = T0[1]
Theta_= T0[5]

xi ,yi,thetai = X0_[0],Y0_[0],Theta_[0]
xf ,yf,thetaf = X0_[-1],Y0_[-1],Theta_[-1]


Xi = [xi,yi,thetai]
Xf = [xf,yf,thetaf]

X,Y,Theta0,V1,W,V2,U1,U2,U3 = MH_DOC(1/7,1/7+2/21,1/7,1/7,1/21,1/7,1/7,Xi,Xf)


cl = []
cu = []

for i in range (6*(n+2)+1):
    cl.append(0)
    cu.append(0)


# In[ ]:


sigma = [10**(-5),5*10**(-5),10**(-4),5*10**(-4),10**(-3),5*10**(-3),10**(-2),5*10**(-2),10**(-1),5*10**(-1),10**(-0)]


for mm in range (len(sigma)):
    X0 = X + np.random.normal(0,sigma[mm],size= (n))
    X0[0], X0[-1] = X[0],X[-1]
    Y0 = Y + np.random.normal(0,sigma[mm],size= (n))
    Y0[0], Y0[-1] = Y0[0],Y0[-1]
    Theta = Theta0 + np.random.normal(0,sigma[mm],size= (n))
    Theta[0], Theta[-1] = Theta0[0],Theta0[-1]

    Vx = (vertcat(X0[1:],X0[-1])-X0)/taux
    Vy = (vertcat(Y0[1:],Y0[-1])-Y0)/taux

    V01 = Vx*cos(Theta0) + sin(Theta)*Vy
    V02 = -Vx*sin(Theta) + cos(Theta)*Vy
    W0  = (vertcat(Theta[1:],Theta[-1])-Theta)/taux

    U01 = (vertcat(V01[1:],V01[-1])-V01)/taux
    U02 = (vertcat(W0[1:],W0[-1])-W0)/taux
    U03 = (vertcat(V02[1:],V02[-1])-V02)/taux
    
    
    ########### BL1
    ## BL1D
    x0 = np.zeros(9*n)
    x0[6*n:7*n],x0[7*n:8*n],x0[8*n:],x0[3*n:4*n],x0[4*n:5*n],x0[5*n:6*n],x0[:n],x0[n:2*n],x0[2*n:3*n] = U1,U2,U3,V1,W,V2,X,Y,Theta0
    
    nlp = cyipopt.Problem(n=len(x0),m=len(cl),problem_obj=BL1(X0,Y0,Theta,xi,yi,thetai,xf,yf,thetaf, taux),cl=cl,cu=cu)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    nlp.add_option("acceptable_constr_viol_tol" , 0.0001)
    
    x, info = nlp.solve(x0)
    
    u1_nlp,u2_nlp,u3_nlp,v1_nlp,w_nlp,v2_nlp,x_nlp,y_nlp,theta_nlp = x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n]
    
    ## IKKT 
    c1_bl1,c2_bl1,c3_bl1,c4_bl1,c5_bl1,c6_bl1,c7_bl1 = KKT (x_nlp,y_nlp,theta_nlp,v1_nlp,w_nlp,v2_nlp,u1_nlp,u2_nlp,u3_nlp,Xi,Xf)
    
    X_bl1,Y_bl1,Theta_bl1,V1_bl1,W_bl1,V2_bl1,U1_bl1,U2_bl1,U3_bl1 = MH_DOC(c1_bl1,c2_bl1,c3_bl1,c4_bl1,c5_bl1,c6_bl1,c7_bl1,Xi,Xf)
    
    ########### KKT
    c1_kkt,c2_kkt,c3_kkt,c4_kkt,c5_kkt,c6_kkt,c7_kkt = KKT (X0,Y0,Theta,V01,W0,V02,U01,U02,U03,Xi,Xf)
    
    X_kkt,Y_kkt,Theta_kkt,V1_kkt,W_kkt,V2_kkt,U1_kkt,U2_kkt,U3_kkt = MH_DOC(c1_kkt,c2_kkt,c3_kkt,c4_kkt,c5_kkt,c6_kkt,c7_kkt,Xi,Xf)
    
    ########## BL 
    res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)

    c1_bl,c2_bl,c3_bl,c4_bl,c5_bl,c6_bl,c7_bl = res.x

    X_bl,Y_bl,Theta_bl,V1_bl,W_bl,V2_bl,U1_bl,U2_bl,U3_bl = MH_DOC(c1_bl,c2_bl,c3_bl,c4_bl,c5_bl,c6_bl,c7_bl,Xi,Xf)
    
    
    plt.figure (figsize=(25,25))
    plt.suptitle("les RMSE_plan et RMSE_ang : \n  BL1D+IKKT : ({} m,{} rad) \n IKKT : ({} m, {}rad) \n BL : ({} m, {} rad) "
             .format(sqrt((np.linalg.norm(X-X_bl1)**2 + np.linalg.norm(Y-Y_bl1)**2)/n),sqrt((np.linalg.norm(Theta0-Theta_bl1)**2)/n),sqrt((np.linalg.norm(X-X_kkt)**2 + np.linalg.norm(Y-Y_kkt)**2)/n),sqrt((np.linalg.norm(Theta0-Theta_kkt)**2)/n),sqrt((np.linalg.norm(X-X_bl)**2 + np.linalg.norm(Y-Y_bl)**2)/n),sqrt((np.linalg.norm(Theta0-Theta_bl)**2)/n)))
    plt.subplot(2,2,1)
    plt.plot(X0,Y0,'*',label = "Trajectoire bruité avec sigma = {}".format(sigma[mm]))
    plt.plot(X,Y,label = 'Trajectoire initial')
    plt.plot(X_bl1,Y_bl1,label = 'DOC (BL1D+IKKT)')
    plt.plot(X_kkt,Y_kkt,label = 'DOC (IKKT)')
    plt.plot(X_bl,Y_bl,label = 'DOC (BL)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    
    
    plt.subplot(2,2,2)
    plt.plot(T,X0,'*',label = "Trajectoire bruité avec sigma = {}".format(sigma[mm]))
    plt.plot(T,X,label = 'Trajectoire initial')
    plt.plot(T,X_bl1,label = 'DOC (BL1D+IKKT)')
    plt.plot(T,X_kkt,label = 'DOC (IKKT)')
    plt.plot(T,X_bl,label = 'DOC (BL)')
    plt.xlabel('Time [s]')
    plt.ylabel('X [m]')
    
    
    plt.subplot(2,2,3)
    plt.plot(T,Y0,'*',label = "Trajectoire bruité avec sigma = {}".format(sigma[mm]))
    plt.plot(T,Y,label = 'Trajectoire initial')
    plt.plot(T,Y_bl1,label = 'DOC (BL1D+IKKT)')
    plt.plot(T,Y_kkt,label = 'DOC (IKKT)')
    plt.plot(T,Y_bl,label = 'DOC (BL)')
    plt.xlabel('Time [s]')
    plt.ylabel('Y [m]')  
    
    
    plt.subplot(2,2,4)
    plt.plot(T,Theta,'*',label = "Trajectoire bruité avec sigma = {}".format(sigma[mm]))
    plt.plot(T,Theta0,label = 'Trajectoire initial')
    plt.plot(T,Theta_bl1,label = 'DOC (BL1D+IKKT)')
    plt.plot(T,Theta_kkt,label = 'DOC (IKKT)')
    plt.plot(T,Theta_bl,label = 'DOC (BL)')
    plt.xlabel('Time [s]')
    plt.ylabel('Theta [m]')
                
    plt.savefig('bruit_sigma={}.png'.format(sigma[mm]))
            