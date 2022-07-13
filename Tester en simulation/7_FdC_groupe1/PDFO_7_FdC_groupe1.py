import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *




def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 

def orientation (x,y,theta, r, i):
    X00 = [x,x+r*cos(theta)]
    Y00 = [y,y+r*sin(theta)]
    
    
    if i == 1 :
        plt.plot(X00,Y00,'r',label='Orientation')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'red' )
        plt.legend()
    else :
        plt.plot(X00,Y00,'r')
        plt.arrow(X00[-1],Y00[-1], 0.001*cos(theta),0.001*sin(theta), width = 0.002, color = 'red' )
        

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r




m01 ,m02 = 62.60648207785031 ,8716.76547379362
m11 ,m12 = 0.0 ,1154653828.5970802
m21 ,m22 = 20.543893994759976 ,7861.558309796407
m31 ,m32 = 488.0082137605221 ,489.1307329027765
m41 ,m42 = 0.003127686679216384 ,0.15123248841357761
m51 ,m52 = 41.753853472012764 ,73.75947511558543
m61 ,m62 = 0.0 ,115465.38285970801



n = 500
taux = 5/n
T = linspace(0,5,n)



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
for i in range (1):
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



options = {'maxfev': 10000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-8}

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
    
    
    opti.minimize(  taux*( c1 *((dot(u1,u1)-m01 )/(m02-m01))+ c2*((dot(u2,u2 )-m11 )/(m12-m11))+ c3*((dot(u3 ,u3)-m21)/(m22-m21)) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m31 )/(m32-m31))+ c5 * ((Max(x,y,Xf[0],Xf[1])-m41 )/(m42-m41)) + c6* (((dot(v1,v1)+ dot(v2,v2))-m51)/(m52-m51)) + c7* ((dot(w,w)-m61)/(m62-m61)) ) )     
    
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
    
    m001 = sqrt((np.linalg.norm(Xmoy-X1_1)**2 + np.linalg.norm(Ymoy-X2_1)**2 + np.linalg.norm(Theta_moy-X3_1)**2 )/n)
    m002 = 10* np.abs (c1 + c2 + c3 + c4 + c5 + c6 + c7 -1)
    m003 = 10* mk
    
    m1 = float(m001+m002+m003)

    return m1
    




T0 = np.loadtxt('E1500.dat')
X = T0[0]
Y = T0[1]
Theta = T0[5]
Xi = [X[0], Y[0],Theta[0]]
Xf = [X[-1], Y[-1],Theta[-1]]

c1 = 0.25
c2 = 0.15
c3 = 0.35
c4 = 0.05
c5 = 0.05
c6 = 0.05
c7 = 0.1
Xmoy,Ymoy,Theta_moy,v1,w,v2,u1,u2,u3 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)




res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)



c1,c2,c3,c4,c5,c6,c7 = res.x

x,y,o,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)




plt.figure (figsize = (20,15))
plt.title("RMSE_plan = {}, RMSE_ang_rad = {}".format(sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 )/n),sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)))
plt.plot(Xmoy,Ymoy,'r',label = 'Trajectoire initial')
plt.plot(x,y,'green',label = 'PDFO')

orientation (Xmoy[0],Ymoy[0],Theta_moy[0] , 0.005, 0)
orientation (Xmoy[100],Ymoy[100],Theta_moy[100] , 0.005, 0)
orientation (Xmoy[200],Ymoy[200],Theta_moy[200] , 0.005, 0)
orientation (Xmoy[300],Ymoy[300],Theta_moy[300] , 0.005, 0)
orientation (Xmoy[400],Ymoy[400],Theta_moy[400] , 0.005, 0)
orientation (Xmoy[-1],Ymoy[-1],Theta_moy[-1] , 0.005, 1)
plt.xlabel("X[m]")
plt.ylabel("Y[m]")
plt.legend()
plt.savefig('PDFO_7FdC_essai.png')

T0 = np.loadtxt('E1500.dat')
Xmoy = T0[0]
Ymoy = T0[1]
Theta_moy = T0[5]
Xi = [X[0], Y[0],Theta[0]]
Xf = [X[-1], Y[-1],Theta[-1]]

res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)



c1,c2,c3,c4,c5,c6,c7 = res.x

x,y,o,v01,w0,v02,u01,u02,u03 = MH_DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf)




plt.figure (figsize = (20,15))
plt.title("RMSE_plan = {}, RMSE_ang_rad = {}".format(sqrt((np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 )/n),sqrt((np.linalg.norm(Theta_moy-o)**2 )/n)))
plt.plot(Xmoy,Ymoy,'r',label = 'Trajectoire initial')
plt.plot(x,y,'green',label = 'PDFO')

orientation (Xmoy[0],Ymoy[0],Theta_moy[0] , 0.005, 0)
orientation (Xmoy[100],Ymoy[100],Theta_moy[100] , 0.005, 0)
orientation (Xmoy[200],Ymoy[200],Theta_moy[200] , 0.005, 0)
orientation (Xmoy[300],Ymoy[300],Theta_moy[300] , 0.005, 0)
orientation (Xmoy[400],Ymoy[400],Theta_moy[400] , 0.005, 0)
orientation (Xmoy[-1],Ymoy[-1],Theta_moy[-1] , 0.005, 1)
plt.xlabel("X[m]")
plt.ylabel("Y[m]")
plt.legend()
plt.savefig('PDFO_7FdC_traj_moy.png')