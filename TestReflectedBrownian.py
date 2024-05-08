# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:21:57 2024

@author: Vviik
"""

import numpy as np
import matplotlib.pyplot as plt
import statistics

number_of_particles = int(1e4)
number_of_timesteps = int(1e3)
x0 = np.random.rand(number_of_particles)
dt = 0.1
v = 1
xlin = np.linspace(0,10,50)
ylin = np.linspace(-10,10,50)
def gaussian(x, mean, dev):
    return 1/(dev*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mean)/dev)**2)

def singlebrownianmotion(x0, dt):
    ret=x0+np.random.choice(np.array([-v*dt,v*dt]), size=x0.shape)
    ret=ret/2+np.abs(ret)/2
    return ret

x = np.copy(x0)
y = np.copy(x0)
std = np.empty(number_of_timesteps)
ystd = np.empty(number_of_timesteps)
for j in range(number_of_timesteps):
    x = singlebrownianmotion(x, dt)
    y = y+np.random.choice(np.array([-v*dt,v*dt]), size=y.shape)
    xmean = statistics.mean(x)
    std[j] = statistics.stdev(x, xbar=0)
    ymean = statistics.mean(y)
    ystd[j] = statistics.stdev(y, xbar=ymean)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)    
    fig.suptitle('Particle distribution. Timestep: '+str(j+1))
    ax1.hist(x, bins=int(np.sqrt(number_of_particles/5)), density = True)
    ax1.plot(xlin, 2*gaussian(xlin, 0, std[j]))
    ax1.set_title('Reflection. Std: '+str(std[j])[:3])
    
    ax2.hist(y, bins=int(np.sqrt(number_of_particles/5)), density = True)
    ax2.plot(ylin, gaussian(ylin, ymean, ystd[j]))
    ax2.set_title('No reflection. Std: '+str(ystd[j])[:3])
    
plt.figure()
plt.ylabel('Standard deviation')
plt.xlabel('timestep')
plt.loglog(std, '.', label='Reflected')
plt.loglog(ystd, '.', label='Non-reflected')
plt.legend()

    
