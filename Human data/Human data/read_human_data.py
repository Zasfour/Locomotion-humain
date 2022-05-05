import numpy as np
import matplotlib.pylab as plt

# This file load and plot 3 different human trajectories with different starting 
# positions and orientations :
# - human_traj_1 starts theorically from (-4,-0.9,-pi/2)
# - human_traj_2 starts theorically from (-1.5,1.2,0)
# - human_traj_3 starts theorically from (-4,-3.4,pi/2)
# The goal position is always (0,0,pi/2).

# The file are organized as followed :
# - from 0 to 5 : mean trajectory (0->x,1->y,2->vx,3->vy,4->theta local,5->theta global)
# - from 6 to 11: trajectory of suject 1
# - from 12 to 17: trajectory of suject 2
# - from 18 to 23: trajectory of suject 3
# - from 24 to 29: trajectory of suject 4
# - from 30 to 35: trajectory of suject 5
# - from 36 to 41: trajectory of suject 6
# - from 42 to 47: trajectory of suject 7
# - from 48 to 53: trajectory of suject 8
# - from 54 to 59: trajectory of suject 9
# - from 60 to end: trajectory of suject 10
# /!\ some individual trajectory have no measured theta, they are not taken into 
# account for the mean theta

# All trajectories counts 500 points

def plot_traj(name,ind):
	plt.subplot(3,3,ind*3+1)
	m = np.loadtxt(name)
	plt.plot(m[0],m[1],color='green',label='mean traj')
	plt.plot(m[6],m[7],color='green',linewidth=0.75,alpha = 0.5,label='individual traj')
	plt.plot(m[12],m[13],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[18],m[19],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[24],m[25],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[30],m[31],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[36],m[37],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[42],m[43],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[48],m[49],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[54],m[55],color='green',linewidth=0.75,alpha = 0.5)
	plt.plot(m[60],m[61],color='green',linewidth=0.75,alpha = 0.5)
	plt.ylabel("y (m)")
	plt.xlabel("x (m)")
	plt.legend()


	arrow_len = 0.1
	count = 0
	for i in range (len(m[0])):
		if count%10 == 0:
			plt.arrow(m[0][i], m[1][i], np.cos(m[5][i])*arrow_len, np.sin(m[5][i])*arrow_len, head_width=.005)
			if np.sum(m[11]) != 0:
				plt.arrow(m[6][i], m[7][i], np.cos(m[11][i])*arrow_len, np.sin(m[11][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[17]) != 0:		
				plt.arrow(m[12][i], m[13][i], np.cos(m[17][i])*arrow_len, np.sin(m[17][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[23]) != 0:			
				plt.arrow(m[18][i], m[19][i], np.cos(m[23][i])*arrow_len, np.sin(m[23][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[29]) != 0:			
				plt.arrow(m[24][i], m[25][i], np.cos(m[29][i])*arrow_len, np.sin(m[29][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[35]) != 0:			
				plt.arrow(m[30][i], m[31][i], np.cos(m[35][i])*arrow_len, np.sin(m[35][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[41]) != 0:			
				plt.arrow(m[36][i], m[37][i], np.cos(m[41][i])*arrow_len, np.sin(m[41][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[47]) != 0:			
				plt.arrow(m[42][i], m[43][i], np.cos(m[47][i])*arrow_len, np.sin(m[47][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[53]) != 0:			
				plt.arrow(m[48][i], m[49][i], np.cos(m[53][i])*arrow_len, np.sin(m[53][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[59]) != 0:			
				plt.arrow(m[54][i], m[55][i], np.cos(m[59][i])*arrow_len, np.sin(m[59][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
			if np.sum(m[65]) != 0:			
				plt.arrow(m[60][i], m[61][i], np.cos(m[65][i])*arrow_len, np.sin(m[65][i])*arrow_len, head_width=.005,color='red',alpha = 0.3)
		count += 1


	# Speed
	plt.subplot(3,3,ind*3+2)
	plt.plot(time,m[0],label='x')
	plt.plot(time,m[1],label='y')
	plt.plot(time,m[2],label='vx')
	plt.plot(time,m[3],label='vy')
	plt.xlabel("time")
	plt.legend()

	# Orientation
	plt.subplot(3,3,ind*3+3)
	plt.plot(time,m[4],label='theta local')
	plt.plot(time,m[5],label='theta global')
	plt.xlabel("time")
	plt.legend()


end = 100
time = np.arange(0,end,0.2)
list_name = ["human_traj_1.dat","human_traj_2.dat","human_traj_3.dat"]

for i in range(len(list_name)):
	plot_traj(list_name[i], i)

plt.show()	