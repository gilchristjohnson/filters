import numpy as np
from numba import float64
from numba.experimental import jitclass

@jitclass([('Vk', float64[:])])
class Lidar:
    def __init__(self, Vk=np.array([0,0], dtype=np.float64)):

        """
        Implimentation of 2-D LiDAR Based Sensor Model. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        """

        self.Vk = Vk

        return
    
    def update(self, observation=np.array([0,0], dtype=np.float64)):
        
        """Update function which is called when a new state estimate is received."""

        Zk = np.array([observation[0]*np.cos(observation[1]),
                       observation[0]*np.sin(observation[1])])

        Wk = np.array([np.sqrt((np.cos(observation[1]) * self.Vk[0])**2 + (observation[0] * np.sin(observation[1]) * self.Vk[1])**2),
                       np.sqrt((np.sin(observation[1]) * self.Vk[0])**2 + (observation[0] * np.cos(observation[1]) * self.Vk[1])**2)])       

                 
        return Zk, Wk