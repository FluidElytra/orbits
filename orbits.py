# Ajouter une classe fusée, sensible à tous les champs gravitationnel
# Programmer la trajectoire de la fusée
# Apprentissage pour faire atterrir la fusée sur mars (TensorFlow?)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# from scipy import io as sio
import math
from pylab import *

# ------------------------------
# CLASSES
# ------------------------------
class body:
	"""A planet"""
	def __init__(self, N_t):
		self.name = "planet"
		self.x  = np.zeros((int(N_t), 1), dtype=float)
		self.y  = np.zeros((int(N_t), 1), dtype=float)
		self.vx = np.zeros((int(N_t), 1), dtype=float)
		self.vy = np.zeros((int(N_t), 1), dtype=float)
		self.ax = np.zeros((int(N_t), 1), dtype=float)
		self.ay = np.zeros((int(N_t), 1), dtype=float)
		self.m  = 0.0
	def euler(self, x_s, y_s, m_s, dt):
		r      			 = math.sqrt((self.x[t-1, 0]-x_s)**2+(self.y[t-1, 0]-y_s)**2)
		F      			 = -(G*self.m*m_s)/(r**2)
		self.ax[t,0]     = F/self.m*(self.x[t-1,0]-x_s)/r
		self.ay[t,0]     = F/self.m*(self.y[t-1,0]-y_s)/r
		self.vx[t,0]     = self.vx[t-1,0] + dt*self.ax[t-1,0]
		self.vy[t,0]     = self.vy[t-1,0] + dt*self.ay[t-1,0]
		self.x[t,0]  	 = self.x[t-1,0]  + dt*self.vx[t-1,0]
		self.y[t,0]      = self.y[t-1,0]  + dt*self.vy[t-1,0]
	def verlet(self, x_s, y_s, m_s, dt):
		r      			 = math.sqrt((self.x[t-1, 0]-x_s)**2+(self.y[t-1, 0]-y_s)**2)
		F      			 = -(G*self.m*m_s)/(r**2)
		ax0    			 = F/self.m*(self.x[t-1,0]-x_s)/r
		ay0			     = F/self.m*(self.y[t-1,0]-y_s)/r
		self.x[t,0]  	 = self.x[t-1,0] + dt*self.vx[t-1,0] + 0.5*dt**2*ax0
		self.y[t,0]  	 = self.y[t-1,0] + dt*self.vy[t-1,0] + 0.5*dt**2*ay0
		r      			 = math.sqrt((self.x[t, 0]-x_s)**2+(self.y[t, 0]-y_s)**2)
		F      			 = -(G*self.m*m_s)/(r**2)
		self.ax[t,0]     = F/self.m*(self.x[t,0]-x_s)/r
		self.ay[t,0]     = F/self.m*(self.y[t,0]-y_s)/r
		self.vx[t,0]	 = self.vx[t-1,0] + 0.5*dt*(ax0+self.ax[t,0])
		self.vy[t,0]	 = self.vy[t-1,0] + 0.5*dt*(ay0+self.ay[t,0])

# ------------------------------
# CONSTANTES
# ------------------------------
# Numerical constants
N_t 	 	  = 100000								# [-] 		 Iteration number
dt  	 	  = 3600.0*24.0							# [s] 		 Time step (1 day)
ua  	 	  = 149597870700						# [m] 		 Unité astronomique (distance Terre-Soleil)
um 			  = 1.989e30							# [kg]		 Unité massique (masse du Soleil)
me			  = 5.972e24							# [kg]		 Masse terrestre
# Physic constants
G   	 	  = 6.67408e-11 						# [m3/kg/s2] Gravitational constant
# Planets initialization
earth1        = body(N_t)
earth1.m  	  = 100*me
earth1.x[0,0] = ua/2

earth2  	  = body(N_t)
earth2.m  	  = me
earth2.x[0,0] = -ua/2	

earth1.vy[0,0]= -math.sqrt(-G*earth2.m*earth2.x[0,0]/(-earth2.x[0,0]+earth1.x[0,0])**2)
earth2.vy[0,0]= math.sqrt(G*earth1.m*earth1.x[0,0]/(-earth2.x[0,0]+earth1.x[0,0])**2)

# ------------------------------
# CALCULS DE TRAJECTOIRE
# ------------------------------
for t in range(1, int(N_t)):
	earth1.verlet(earth2.x[t-1,0], earth2.y[t-1,0], earth2.m, dt)
	earth2.verlet(earth1.x[t-1,0], earth1.y[t-1,0], earth1.m, dt)

# ------------------------------
# ANIMATION
# ------------------------------
plt.style.use('dark_background')
fig = plt.figure()
ax = plt.axes(xlim=(-1, 1), ylim=(-1, 6))

plt.xlabel('x [ua]')
plt.ylabel('y [ua]')
planet1, = plt.plot([], [], 'bo')
planet2, = plt.plot([], [], 'ro')

def init():
	planet1.set_data([], [])
	planet2.set_data([], [])
	return planet1, planet2
def animate(i):
	x = earth1.x[50*(i),0]/ua
	y = earth1.y[50*(i),0]/ua
	planet1.set_data(x, y)
	x = earth2.x[50*(i),0]/ua
	y = earth2.y[50*(i),0]/ua
	planet2.set_data(x, y)
	return planet1, planet2

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=int(N_t), interval=1, blit=True)
plt.show()

# ------------------------------
# AFFICHAGE
# ------------------------------
plt.style.use('dark_background')

subplot(121)
plt.plot([x_e / ua for x_e in earth1.x], [y_e / ua for y_e in earth1.y], color="blue", label="body 1")
plt.plot([x_e / ua for x_e in earth2.x], [y_e / ua for y_e in earth2.y], color="red", label="body 2")
plt.legend(loc='upper left')
plt.axis('equal')
plt.xlabel('x [ua]')
plt.ylabel('y [ua]')

subplot(222)
t = np.arange(0.0,dt*len(earth1.vy),dt)
plt.plot([t_e / (dt) for t_e in t], [y_e / 1e-3 for y_e in earth1.vx], color="blue", label="body 1")
plt.plot([t_e / (dt) for t_e in t], [y_e / 1e-3 for y_e in earth2.vx], color="red", label="body 2")
plt.legend(loc='upper left')
plt.xlabel('t [days]')
plt.ylabel('v_x [km/s]')

subplot(224)
t = np.arange(0.0,dt*len(earth1.vy),dt)
plt.plot([t_e / (dt) for t_e in t], [y_e / 1e-3 for y_e in earth1.vy], color="blue", label="body 1")
plt.plot([t_e / (dt) for t_e in t], [y_e / 1e-3 for y_e in earth2.vy], color="red", label="body 2")
plt.legend(loc='upper left')
plt.xlabel('t [days]')
plt.ylabel('v_y [km/s]')

plt.show()
