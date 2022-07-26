import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *
from scipy.optimize import approx_fprime
import cyipopt



def Orientation_ (x,y,theta, m, couleur ):
    plt.arrow(x, y, (m/2)*cos(theta),(m/2)*sin(theta), color = couleur, head_width = m/2, head_length = m)



def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r



Groupe3 = ['O4000.dat','S4000.dat','E4000.dat','S4015.dat','N4000.dat','O4015.dat','E4015.dat','N4015.dat']



m01 ,m02 = 199.48711527253298 ,11291.1993561176
m11 ,m12 = 0.0 ,1014090692.5190746
m21 ,m22 = 58.187911897949476 ,36298.72854919096
m31 ,m32 = 485.2884817929601 ,487.5864338674093
m41 ,m42 = 8.223229991153927e-06 ,0.24546796489495004
m51 ,m52 = 222.6716365422084 ,468.2557813700287
m61 ,m62 = 0.0 ,101409.06925190745




n = 51
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

def interpolation(X):
    x00 = np.zeros(51)
    for i in range (50):
        x00[i] = X[i*10]
    x00[-1] = X[-1]
    return x00

color = ['orangered', 'olive', 'skyblue','firebrick']

cl = []
cu = []

for i in range (6*(n+2)+1):
    cl.append(0)
    cu.append(0)

x0 = np.zeros(9*n)



for j in range (len(Groupe3)):

    T0 = np.loadtxt(Groupe3[j])
    X0 = interpolation(T0[0])
    Y0 = interpolation(T0[1])
    Theta = interpolation(T0[5])

    xi = X0[0]
    yi = Y0[0]
    thetai = Theta[0]
    xf = X0[-1]
    yf = Y0[-1]
    thetaf = Theta[-1]
    
    Xi = [xi,yi,thetai]
    Xf = [xf,yf,thetaf]

    Vx = (vertcat(X0[1:],X0[-1])-X0)/taux
    Vy = (vertcat(Y0[1:],Y0[-1])-Y0)/taux
    
    V1 = Vx*cos(Theta)+Vy*sin(Theta)
    V2 = -Vx*sin(Theta)+Vy*cos(Theta)
    W = (vertcat(Theta[1:],Theta[-1])-Theta)/taux

    U1 = (vertcat(V1[1:],V1[-1])-V1)/taux
    U2 = (vertcat(W[1:],W[-1])-W)/taux
    U3 = (vertcat(V2[1:],V2[-1])-V2)/taux

    xx,y,o,v1,w,v2,u1,u2,u3 = MH_DOC(1/7,1/7,1/7,1/7,1/7,1/7,1/7,Xi,Xf)

    x0 = np.zeros(9*n)
    for i in range (n):
        x0[i] = xx[i]
        x0[i+n] = y[i]
        x0[i+2*n] = o[i]
        x0[i+3*n] = v1[i]
        x0[i+4*n] = w[i]
        x0[i+5*n] = v2[i]
        x0[i+6*n] = u1[i]
        x0[i+7*n] = u2[i]
        x0[i+8*n] = u3[i]
    
    nlp = cyipopt.Problem(n=len(x0),m=len(cl),problem_obj=BL1(X0,Y0,Theta,xi,yi,thetai,xf,yf,thetaf, taux),cl=cl,cu=cu)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    nlp.add_option("acceptable_constr_viol_tol" , 0.0001)

    x, info = nlp.solve(x0)

    u1_nlp,u2_nlp,u3_nlp,v1_nlp,w_nlp,v2_nlp,x_nlp,y_nlp,theta_nlp = x[6*n:7*n],x[7*n:8*n],x[8*n:],x[3*n:4*n],x[4*n:5*n],x[5*n:6*n],x[:n],x[n:2*n],x[2*n:3*n]

    c1,c2,c3,c4,c5,c6,c7 = KKT (x_nlp,y_nlp,theta_nlp,v1_nlp,w_nlp,v2_nlp,u1_nlp,u2_nlp,u3_nlp,Xi,Xf)

    X_BL1,Y_BL1,Theta_BL1,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)

    ######## IKKT

    c1,c2,c3,c4,c5,c6,c7 = KKT (X0,Y0,Theta,V1,W,V2,U1,U2,U3,Xi,Xf)


    X_IKKT,Y_IKKT,Theta_IKKT,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)

    ####### BL

    res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)

    c1,c2,c3,c4,c5,c6,c7 = res.x

    X_BL,Y_BL,Theta_BL,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)

    
    m = (((x[0]-x[-1])**2 + (y[0]-y[-1])**2 )/n)
    m1 = (((X0[0]-X0[-1])**2 + (Y0[0]-Y0[-1])**2 )/n)

    if m1 < 1 :
        m = m*5
        m1 = m1*5

    plt.figure(figsize=(25,25))
    plt.subplot(2,2,1)
    Orientation_ (X_BL1[0],Y_BL1[0],Theta_BL1[0] ,m, color[0])
    Orientation_  (X_BL1[10],Y_BL1[10],Theta_BL1[10] ,m,color[0])
    Orientation_ (X_BL1[20],Y_BL1[20],Theta_BL1[20],m, color[0])
    Orientation_ (X_BL1[30],Y_BL1[30],Theta_BL1[30] ,m, color[0])
    Orientation_ (X_BL1[40],Y_BL1[40],Theta_BL1[40] ,m, color[0])
    Orientation_ (X_BL1[-1],Y_BL1[-1],Theta_BL1[-1], m, color[0])

    Orientation_ (X_BL[0],Y_BL[0],Theta_BL[0] ,m, color[1])
    Orientation_  (X_BL[10],Y_BL[10],Theta_BL[10] ,m,color[1])
    Orientation_ (X_BL[20],Y_BL[20],Theta_BL[20],m, color[1])
    Orientation_ (X_BL[30],Y_BL[30],Theta_BL[30] ,m, color[1])
    Orientation_ (X_BL[40],Y_BL[40],Theta_BL[40] ,m, color[1])
    Orientation_ (X_BL[-1],Y_BL[-1],Theta_BL[-1], m, color[1])

    Orientation_ (X_IKKT[0],Y_IKKT[0],Theta_IKKT[0] ,m, color[2])
    Orientation_  (X_IKKT[10],Y_IKKT[10],Theta_IKKT[10] ,m,color[2])
    Orientation_ (X_IKKT[20],Y_IKKT[20],Theta_IKKT[20],m, color[2])
    Orientation_ (X_IKKT[30],Y_IKKT[30],Theta_IKKT[30] ,m, color[2])
    Orientation_ (X_IKKT[40],Y_IKKT[40],Theta_IKKT[40] ,m, color[2])
    Orientation_ (X_IKKT[-1],Y_IKKT[-1],Theta_IKKT[-1], m, color[2])

    Orientation_ (X0[0],Y0[0],Theta[0] ,m1, color[3])
    Orientation_ (X0[10],Y0[10],Theta[10] ,m1, color[3])
    Orientation_ (X0[20],Y0[20],Theta[20] ,m1, color[3])
    Orientation_ (X0[30],Y0[30],Theta[30] ,m1, color[3])
    Orientation_ (X0[40],Y0[40],Theta[40] ,m1, color[3])
    Orientation_ (X0[-1],Y0[-1],Theta[-1] ,m1, color[3])


    plt.plot(X_BL1,Y_BL1,'greenyellow', label = 'BL1D+IKKT')
    plt.plot(X_BL,Y_BL,'orange', label = 'BL')
    plt.plot(X_IKKT,Y_IKKT,'dodgerblue', label = 'IKKT')
    plt.plot(X0,Y0,'maroon', label = 'mean trajectory by {}'.format(Groupe3[j]))

    plt.plot(x[0],y[0],'o', color = 'tab:green', label = 'initial position')
    plt.plot(x[-1],y[-1],'o', color = 'darkorange', label = 'goal position')
    plt.suptitle("les RMSE_plan et RMSE_ang : \n  BL1D+IKKT : ({} m,{} rad) \n IKKT : ({} m, {}rad) \n BL : ({} m, {} rad) "
             .format(sqrt((np.linalg.norm(X0-X_BL1)**2 + np.linalg.norm(Y0-Y_BL1)**2)/n),sqrt((np.linalg.norm(Theta-Theta_BL1)**2)/n),sqrt((np.linalg.norm(X0-X_IKKT)**2 + np.linalg.norm(Y0-Y_IKKT)**2)/n),sqrt((np.linalg.norm(Theta-Theta_IKKT)**2)/n),sqrt((np.linalg.norm(X0-X_BL)**2 + np.linalg.norm(Y0-Y_BL)**2)/n),sqrt((np.linalg.norm(Theta-Theta_BL)**2)/n)))
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.xlabel ("X [m]")
    plt.ylabel ("Y [m]")
    plt.savefig("Comparaison_{}.png".format(Groupe3[j]))