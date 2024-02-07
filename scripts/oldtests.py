#!/usr/bin/python3
import numpy as np
import time
from OldCTRVMotionModel import CTRVMotionModel
from OldLidarSensorModel import LidarSensorModel

MotionModel = CTRVMotionModel()
SensorModel = LidarSensorModel()

parameters = ([(0, 10), (-5, 5), (0, 3.14), (0, 0), (0, 0)])
num_params = len(parameters)

def initialize_particles(parameters, distribution, N):

        particles = np.empty((N, len(parameters)))

        for index in range(len(parameters)):

            if distribution == 'uniform':

                particles[:, index] = np.random.uniform(parameters[index][0], parameters[index][1], size=N)

            elif distribution == 'gaussian':

                particles[:, index] = np.random.normal(parameters[index][0], parameters[index][1], size=N)

            else:

                raise ValueError(distribution + "is not a valid distribution type.")

        weights = np.ones(N)/N

        return particles, weights
    
def prediction(particles, dt=0.1):

    for index in range(5):

        mXkgkmve1, sXkgkmve1 = MotionModel.update(mX_kmve1=particles[index], sX_kmve1=np.array([0, 0, 0, 0, 0]), dt=dt)

        particles[index] = np.random.multivariate_normal(mean=mXkgkmve1, cov=np.diag(sXkgkmve1))

    return particles

def resample(particles, weights):

    probabilities = weights / np.sum(weights)
    index_numbers = np.random.choice(len(particles), size=len(particles),p=probabilities)

    particles = particles[index_numbers, :]
    weights = weights[index_numbers]

    return particles, weights

def correction(particles, weights, observation=np.array([0,0,0], dtype=np.float64)):

    mX_k, sX_k = SensorModel.update(observation)

    distance = np.linalg.norm(particles[:, 0:2] - mX_k[0:2], axis=1)

    weights = (np.max(distance) - distance)**8

    return particles, weights

particles, weights = initialize_particles(parameters, 'uniform', 1000)

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
average = 0

num_tests = 10

for iteration in range(num_tests):
    start = time.time()
    for iteration in range(10000):
        #particles1 = prediction(particles)
        #particles2, weights2 = correction(particles, weights)
        particles3, weights3 = resample(particles, weights)
    end = time.time()

    average += (end-start)

average = average/num_tests

print("Elapsed average (after compilation) = %s" % (average))