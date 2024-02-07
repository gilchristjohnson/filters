#!/usr/bin/python3
import numpy as np
import time
from numba import float32, int64, prange
from MotionModel import CTRV
from SensorModel import Lidar
from numba import jit, njit
from numba.typed import List

@njit(cache=True, parallel=True, fastmath=True)
def initialize_particles(parameters, distribution, N):

    particles = np.empty((N, num_params), dtype=float32)

    for particle in prange(N):

        for index in prange(num_params):

            if distribution == 'uniform':

                particles[particle][index] = np.random.uniform(parameters[index][0], parameters[index][1])

            elif distribution == 'gaussian':

                particles[particle][index] = np.random.normal(parameters[index][0], parameters[index][1])

            else:

                raise ValueError(distribution + "is not a valid distribution type.")

    weights = np.ones(N, dtype=float32)/N

    return particles, weights

@njit(cache=True, parallel=True, fastmath=True)
def prediction(particles, dt=0.1):

    """Prediction Phase of Particle Filter using CTRV Motion Model."""

    sXkmve1=np.array([0,0,0,0,0], dtype=float32)

    for particle in prange(num_particles):

        mXkmve1 = particles[particle]

        if mXkmve1[4] == 0:
            mXkgXkmve1 = np.array([mXkmve1[0]+(mXkmve1[3]*np.cos(mXkmve1[2])*dt),
                                    mXkmve1[1]+(mXkmve1[3]*np.sin(mXkmve1[2])*dt),
                                    mXkmve1[2],
                                    mXkmve1[3],
                                    mXkmve1[4]])
            
            sXkgXkmve1 = np.array([np.sqrt((sXkmve1[0]**2) + ((dt*np.cos(mXkmve1[2]))**2) + ((dt*mXkmve1[4]*np.sin(mXkmve1[2]))**2)),
                                    np.sqrt((sXkmve1[1]**2) + ((dt*np.sin(mXkmve1[2]))**2) + ((dt*mXkmve1[4]*np.cos(mXkmve1[2]))**2)),
                                    sXkmve1[2],
                                    sXkmve1[3],
                                    sXkmve1[4]])
        
        else:   
            mXkgXkmve1 = np.array([mXkmve1[0] + (mXkmve1[3]/mXkmve1[4]) * (np.sin(mXkmve1[4] * dt  + mXkmve1[2]) - np.sin(mXkmve1[2])),
                                    mXkmve1[1] + (mXkmve1[3]/mXkmve1[4]) * (np.cos(mXkmve1[2]) - np.cos(mXkmve1[4] * dt + mXkmve1[2])),
                                    mXkmve1[2] + mXkmve1[4]*dt,
                                    mXkmve1[3],
                                    mXkmve1[4]])
            
            sXkgXkmve1 = np.array([np.sqrt((sXkmve1[0]**2) + ((dt*np.cos(mXkmve1[2]))**2) + ((dt*mXkmve1[4]*np.sin(mXkmve1[2]))**2)),
                                    np.sqrt((sXkmve1[1]**2) + ((dt*np.sin(mXkmve1[2]))**2) + ((dt*mXkmve1[4]*np.cos(mXkmve1[2]))**2)),
                                    sXkmve1[2],
                                    sXkmve1[3],
                                    sXkmve1[4]])
            
        for index in prange(num_params):

            particles[particle][index] = np.random.uniform(mXkgXkmve1[index], sXkgXkmve1[index])  

    return particles

@njit(cache=True, parallel=True, fastmath=True)
def correction(particles, weights, observation=np.array([0,0], dtype=np.float32)):
        
    """Update Phase of Particle Filter using Lidar Sensor Model"""

    mZk = np.array([observation[0]*np.cos(observation[1]),
                    observation[0]*np.sin(observation[1])])
    
    array = particles[:, 0:2] - mZk[0:2]

    distance = np.empty(num_particles, dtype=float32)

    for i in prange(array.shape[1]):
        distance[i] = np.sqrt(array[0, i] * array[0, i] + array[1, i] * array[1, i])

    weights = (np.max(distance) - distance)**4

    return particles, weights

@njit(cache=True, parallel=True, fastmath=True)
def resample(particles, weights):

    probabilities = weights / np.sum(weights)

    index_numbers = np.arange(0, num_particles, 1)
    sample = np.empty(num_particles, dtype=int64)
    new_particles = np.empty((num_particles, num_params), dtype=float32)
    new_weights = np.empty(num_particles, dtype=float32)

    for index in prange(num_particles):

        sample[index] = index_numbers[np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")]

    for index in prange(num_particles):

        new_particles[index] = particles[sample[index]]
        new_weights[index] = weights[sample[index]]

    return new_particles, new_weights

parameters = List([(0, 10), (-5, 5), (0, 3.14), (0, 0), (0, 0)])
num_params = 5
num_particles = 1000
MotionModel = CTRV()
SensorModel = Lidar()

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
particles, weights = initialize_particles(parameters, 'uniform', num_particles)
particles = prediction(particles)
particles2, weights2 = correction(particles, weights)
particles3, weights3 = resample(particles, weights)
#mean, var = estimate(particles,weights)

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
average = 0

num_tests = 10

for iteration in range(num_tests):
    start = time.time()
    for iteration in range(10000):
        #particles1 = prediction(particles)
        #particles2, weights2 = correction(particles, weights)
        particles3, weights3 = resample(particles, weights)
        #mean, var = estimate(particles,weights)
    end = time.time()

    average += (end-start)

average = average/num_tests

print("Elapsed average (after compilation) = %s" % (average))