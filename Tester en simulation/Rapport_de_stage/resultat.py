#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as random
from pdfo import *
from casadi import *



def plot_traj(name,ind,X,Y,O):
    plt.subplot(1,4,ind+1)
    mm0 = np.loadtxt(name)
    mn = ((mm0[0][0]-mm0[0][-1])**2 + (mm0[1][0]-mm0[1][-1])**2)**(1/2)/30
    x = mm0[0]
    y = mm0[1]
    o = mm0[5]
    Orientation_ (x[0],y[0],o[0] ,mn, 'olive')
    Orientation_ (x[50],y[50],o[50] ,mn, 'olive')
    Orientation_ (x[100],y[100],o[100] ,mn,'olive')
    Orientation_ (x[150],y[150],o[150] ,mn, 'olive')
    Orientation_ (x[200],y[200],o[200] ,mn, 'olive')
    Orientation_ (x[250],y[250],o[250] ,mn, 'olive')
    Orientation_ (x[300],y[300],o[300] ,mn, 'olive')
    Orientation_ (x[350],y[350],o[350] ,mn, 'olive')
    Orientation_ (x[400],y[400],o[400] ,mn, 'olive')
    Orientation_ (x[450],y[450],o[450] ,mn, 'olive')
    Orientation_ (x[-1],y[-1],o[-1], mn, 'olive')
    
    Orientation_ (X[0],Y[0],O[0] ,mn, 'orangered')
    Orientation_ (X[50],Y[50],O[50] ,mn, 'orangered')
    Orientation_ (X[100],Y[100],O[100] ,mn,'orangered')
    Orientation_ (X[150],Y[150],O[150] ,mn, 'orangered')
    Orientation_ (X[200],Y[200],O[200] ,mn, 'orangered')
    Orientation_ (X[250],Y[250],O[250] ,mn, 'orangered')
    Orientation_ (X[300],Y[300],O[300] ,mn, 'orangered')
    Orientation_ (X[350],Y[350],O[350] ,mn, 'orangered')
    Orientation_ (X[400],Y[400],O[400] ,mn, 'orangered')
    Orientation_ (X[450],Y[450],O[450] ,mn, 'orangered')
    Orientation_ (X[-1],Y[-1],O[-1], mn, 'orangered')
    
    plt.plot(X,Y,color='red',label='DOC (BL)')
    plt.plot(mm0[0],mm0[1],color='green',label='Trajectoire moyenne')
    plt.plot(mm0[6],mm0[7],color='green',linewidth=0.5,alpha = 0.5,label='Trajectoire individuelle')
    plt.plot(mm0[12],mm0[13],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[18],mm0[19],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[24],mm0[25],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[30],mm0[31],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[36],mm0[37],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[42],mm0[43],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[48],mm0[49],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[54],mm0[55],color='green',linewidth=0.5,alpha = 0.5)
    plt.plot(mm0[60],mm0[61],color='green',linewidth=0.5,alpha = 0.5)
    #ax = plt.gca()
    #ax.set_aspect("equal")
    plt.grid()
    
    plt.ylabel("y [m]")
    plt.xlabel("x [m]")
    plt.legend()



def Orientation_ (x,y,theta, m, couleur) :
    plt. arrow(x, y, (m/2)*cos(theta),(m/2)*sin(theta), color = couleur, head_width = m/2, head_length = m)


def tracer_orientation (x,y,theta, r, i):
    if i == 1 :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' , label = "Axe local suivant x")
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' , label = "Axe local suivant y")
        plt.legend()
    else :
        plt.arrow(x, y, r*cos(theta),r*sin(theta), width = 0.01, color = 'red' )
        plt.arrow(x, y, r*cos(pi/2+theta),r*sin(pi/2+theta), width = 0.01, color = 'yellow' )
 


def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum());
    return r



gp1 = ['E1500.dat','S1500.dat','O1500.dat','N1500.dat']
m1 = [62.60648207785031 , 8716.76547379362 , 0.0 ,1154653828.5970802 , 20.543893994759976 ,7861.558309796407 , 488.0082137605221 ,489.1307329027765 , 0.003127686679216384 ,0.15123248841357761 , 41.753853472012764 ,73.75947511558543 ,  0.0 ,115465.38285970801]

gp2 = ['O0615.dat','O1515.dat','N0615.dat','S0615.dat','S1515.dat','E0615.dat','N1515.dat','E1515.dat','O-0615.dat','N-0615.dat','S-0615.dat','E-0615.dat']
m2 = [216.6058685341806 ,18770.730081753416 , 0.0 ,91445082.23389618 , 285.12944960426574 ,10778.161111878238 , 486.3454140114936 ,488.23333202168567 , 4.249648425966459e-05 ,0.6546337152297712 , 97.61305051332354 ,264.71837454603855, 0.0 ,9144.508223389617]

gp3 = ['O4000.dat','S4000.dat','E4000.dat','S4015.dat','N4000.dat','O4015.dat','E4015.dat','N4015.dat']
m3 = [199.48711527253298 ,11291.1993561176 , 0.0 ,1014090692.5190746 , 58.187911897949476 ,36298.72854919096 , 485.2884817929601 ,487.5864338674093 , 8.223229991153927e-06 ,0.24546796489495004, 222.6716365422084 ,468.2557813700287,0.0 ,101409.06925190745]

gp4 = ['O0640.dat','O1540.dat','S1540.dat','N0640.dat','S0640.dat','O-0640.dat','N1540.dat','E0640.dat','N-0640.dat','S-0640.dat','E1540.dat','E-0640.dat']
m4 = [279.2985516284672 ,36819.88123745937, 0.0 ,188499082.09456128, 120.88275779074856 ,7313.609086193289, 485.0029590420518 ,486.529022210082, 3.0371617906983566e-06 ,0.0615935494922574, 225.18595968771533 ,596.5164638799275, 0.0 ,18849.908209456127]

gp5 = ['S4040.dat','O4040.dat','N4040.dat','E4040.dat']
m5 = [292.3851250104687 ,178823.9431373308, 0.0 ,14364771.55522439, 157.2849782756893 ,25817.888756336815, 484.7216606041334 ,486.75514687455075,  7.570873092468666e-05 ,0.30737918541277875, 289.2903712539735 ,670.0581822295461, 0.0 ,1436.4771555224393]



n = 500
taux = 5/n
T = linspace(0,5,n)




x = SX.sym('x',n)
y = SX.sym('y',n)
xf = SX.sym('xf',1)
yf = SX.sym('yf',1)


N = SX.zeros(1)
for i in range (n-1):
    N += ((xf - x[i+1])**2 + (yf - y[i+1])**2 + 10**(-5) )/((xf - x[i])**2 + (yf - y[i])**2 + 10**(-5) )
                                                
Direct = Function('Direct', [x,xf,y,yf],[N])


N0 = SX.zeros(1)
for i in range (n-1):
    N0 += fmax(0,(x[i+1]-xf)**2 + (y[i+1]-yf)**2 - ((x[i]-xf)**2 + (y[i]-yf)**2))
    
Max = Function ('Max',[x,y,xf,yf],[N0])



dot_ = SX.zeros(1)
x_prime = (x[1:]-x[:-1])/taux
for i in range (n-1):
    dot_ += (x_prime[i])**2
    
D_F = Function ('D_F',[x],[dot_])



x_prime2 = (x_prime[1:]-x_prime[:-1])/taux
dot_1 = SX.zeros(1)
for i in range (n-2):
    dot_1 += (x_prime2[i])**2
    
D_F_ang = Function ('D_F_ang',[x],[dot_1])




def DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,X,Y,THETA,V1,V2):
    #print(c1,c2,c3,c4,c5,c6,c7)
    
    opti = casadi.Opti()   
    ## les positions
    x = opti.variable(n)
    y = opti.variable(n)
    theta = opti.variable(n)

    ## les vitesses 
    v1 = opti.variable(n)        ## vitesse latérale
    v2 = opti.variable(n)        ## vitesse orthogonal
    
    
    opti.minimize(  taux*( c1 *((D_F(v1)-m[0] )/(m[1]-m[0]))+ c2*((D_F_ang(theta)-m[2] )/(m[3]-m[2]))+ c3*((D_F(v2)-m[4])/(m[5]-m[4])) +  c4 * ((Direct(y,Xf[1],x,Xf[0])-m[6] )/(m[7]-m[6]))+ c5 *((Max(x,y,Xf[0],Xf[1])-m[8] )/(m[9]-m[8]))+ c6* (((dot(v1,v1)+ dot(v2,v2))-m[10])/(m[11]-m[10])) + c7* ((D_F(theta)-m[12])/(m[13]-m[12])) ) )     
    
    opti.subject_to( x[0] == Xi[0] )        
    opti.subject_to( y[0] == Xi[1] )
    opti.subject_to( theta[0] == Xi[2] )
    opti.subject_to( v1[0] == 0 )
    opti.subject_to( v2[0] == 0 )

    ## pour les contraintes d'égaliter
    opti.subject_to( x[1:]  == x[:n-1]+taux*(cos(theta[:n-1])*v1[:n-1] - sin(theta[:n-1])*v2[:n-1]) )
    opti.subject_to( y[1:]  == y[:n-1]+taux*(sin(theta[:n-1])*v1[:n-1] + cos(theta[:n-1])*v2[:n-1]) )

    ## pour les conditions finales
    opti.subject_to( x[-1]== Xf[0] )
    opti.subject_to( y[-1]== Xf[1] )
    opti.subject_to( theta[-1] == Xf[2] )
    opti.subject_to( v1[-1] == 0 ) 
    opti.subject_to( v2[-1] == 0 )
    
    opti.set_initial(x, X)
    opti.set_initial(y, Y)
    opti.set_initial(theta, THETA)
    opti.set_initial(v1, V1)
    opti.set_initial(v2, V2)
    

    opti.solver('ipopt', {"expand" : True}, {"acceptable_constr_viol_tol" : 0.01})             

    sol = opti.solve()
    

    return sol.value(x),sol.value(y),sol.value(theta),sol.value(v1),sol.value(v2)




options = {'maxfev': 100000 , 'rhobeg' : 0.1 , 'rhoend' : 1e-10}

bounds1 = np.array([[0, 1], [0, 1] , [0, 1], [0, 1], [0, 1] , [0, 1], [0, 1]])
lin_con1 = LinearConstraint([1, 1, 1, 1, 1, 1 ,1], 1, 1)




def PDFO(C):
    
    c1,c2,c3,c4,c5,c6,c7 = C
    
    mk = 0
    
    if c1 < 0:
        mk+= -c1
        c1 = 0
    if c2 < 0:
        mk+= -c2
        c2 = 0
    if c3 < 0:
        mk+= -c3
        c3 = 0
    if c4 < 0:
        mk+= -c4
        c4 = 0
    if c5 < 0:
        mk+= -c5
        c5 = 0
    if c6 < 0:
        mk+= -c6
        c6 = 0
    if c7 < 0:
        mk+= -c7
        c7 = 0
        
    
    print(c1,c2,c3,c4,c5,c6,c7)
    
    x,y,o,v01,v02 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)
    
    mm1 = 10*np.abs(c1+c2+c3+c4+c5+c6+c7-1)
    mm2 = 10*mk
    mm3 = (np.linalg.norm(Xmoy-x)**2 + np.linalg.norm(Ymoy-y)**2 + np.linalg.norm(Theta_moy-o)**2)/n
    print(mm3)
        
    return float(mm1 +mm2 +mm3 )




m = m1
plt.figure(figsize = (40,7))

for i in range (4):
    T0 = np.loadtxt(gp1[i])
    Xmoy = T0[0]  
    Ymoy = T0[1]  
    Theta_moy = T0[5]  
    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
    Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
    Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
    Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)
    
    res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)
    
    c1,c2,c3,c4,c5,c6,c7 = res.x
    
    X,Y,O,V1,V2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)
    
    plot_traj(gp1[i],i,X,Y,O)
    
    plt.title('Scénario {}  \n RMSE(x,y) = {} m \n RMSE(theta) = {} rad'.format(i+1, sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))

plt.savefig('PDFO_traj_{}.png'.format(0))  


m = m2
for i in range (3):
    plt.figure(figsize = (40,7))
    for j in range (4):
        T0 = np.loadtxt(gp2[4*i+j])
        Xmoy = T0[0]  
        Ymoy = T0[1]  
        Theta_moy = T0[5]  
        Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
        Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

        Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
        Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
        Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
        Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)

        res = pdfo( PDFO, [1/7,2/7,1/7,0/7,0/7,2/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)

        c1,c2,c3,c4,c5,c6,c7 = res.x

        X,Y,O,V1,V2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)

        plot_traj(gp2[4*i+j],j,X,Y,O)

        plt.title('Scénario {}  \n RMSE(x,y) = {} m \n RMSE(theta) = {} rad'.format(4*i+j+1+4, sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))

    plt.savefig('PDFO_traj_{}.png'.format(i+1))  



m = m3
for i in range (1):
    plt.figure(figsize = (40,7))
    for j in range (4):
        T0 = np.loadtxt(gp3[4*i+j+4])
        Xmoy = T0[0]  
        Ymoy = T0[1]  
        Theta_moy = T0[5]  
        Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
        Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

        Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
        Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
        Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
        Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)

        res = pdfo( PDFO, [1/5,1/5,1/5,0/7,0/7,1/5,1/5],bounds=bounds1, constraints=lin_con1, options=options)

        c1,c2,c3,c4,c5,c6,c7 = res.x

        X,Y,O,V1,V2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)

        plot_traj(gp3[4*i+j+4],j,X,Y,O)

        plt.title('Scénario {}  \n RMSE(x,y) = {} m \n RMSE(theta) = {} rad'.format(4*i+j+1+4+12+4,  sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))

    plt.savefig('PDFO_traj_{}.png'.format(i+4+1))  




m = m4
for i in range (3):
    plt.figure(figsize = (40,7))
    for j in range (4):
        T0 = np.loadtxt(gp4[4*i+j])
        Xmoy = T0[0]  
        Ymoy = T0[1]  
        Theta_moy = T0[5]  
        Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
        Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

        Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
        Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
        Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
        Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)

        res = pdfo( PDFO, [1/2,0/2,0,0,0,1/2,0],bounds=bounds1, constraints=lin_con1, options=options)

        c1,c2,c3,c4,c5,c6,c7 = res.x

        X,Y,O,V1,V2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)

        plot_traj(gp4[4*i+j],j,X,Y,O)

        plt.title('Scénario {}  \n RMSE(x,y) = {} m \n RMSE(theta) = {} rad'.format(4*i+j+1+4+12+8, sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))

    plt.savefig('PDFO_traj_{}.png'.format(i+4+2))  




m = m5
plt.figure(figsize = (40,7))

for i in range (4):
    T0 = np.loadtxt(gp5[i])
    Xmoy = T0[0]  
    Ymoy = T0[1]  
    Theta_moy = T0[5]  
    Xi = [Xmoy[0], Ymoy[0], Theta_moy[0]]
    Xf = [Xmoy[-1], Ymoy[-1], Theta_moy[-1]]

    Vx = (vertcat(Xmoy[1:],Xmoy[-1]) - Xmoy)/taux
    Vy = (vertcat(Ymoy[1:],Ymoy[-1]) - Ymoy)/taux
    Vf = Vx * cos(Theta_moy) + Vy * sin(Theta_moy)
    Vo = -Vx * sin(Theta_moy) + Vy * cos(Theta_moy)
    
    res = pdfo( PDFO, [1/7,1/7,1/7,1/7,1/7,1/7,1/7],bounds=bounds1, constraints=lin_con1, options=options)
    
    c1,c2,c3,c4,c5,c6,c7 = res.x
    
    X,Y,O,V1,V2 = DOC(c1,c2,c3,c4,c5,c6,c7,Xi,Xf,m,Xmoy,Ymoy,Theta_moy,Vf,Vo)
    
    plot_traj(gp5[i],i,X,Y,O)
    
    plt.title('Scénario {}  \n RMSE(x,y) = {} m \n RMSE(theta) = {} rad'.format(i+37,  sqrt((np.linalg.norm(Xmoy-X)**2+np.linalg.norm(Ymoy-Y)**2)/n),sqrt((np.linalg.norm(Theta_moy-O)**2)/n)))

plt.savefig('PDFO_traj_{}.png'.format(9))  