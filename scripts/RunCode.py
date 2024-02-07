#!/usr/bin/python3
import numpy as np
import time
from numba import float64
from vehicular_ws.src.filters.scripts.MotionModel import CTRV
from numba import jit, njit
from numba import float64, int64
from numba.typed import List

MotionModel = CTRV()

parameters = List([(0, 10), (-5, 5), (0, 3.14), (0, 0), (0, 0)])
num_params = len(parameters)
num_particles = 1000

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
particles, weights = initialize_particles(parameters, 'uniform', 1000)
particles = prediction(particles)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
for iteration in range(1000000):
    particles = prediction(particles)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))
