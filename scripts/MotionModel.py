import numpy as np
from numba import float64
from numba.experimental import jitclass

@jitclass([('Wk', float64[:])])
class CTRV:
    def __init__(self, Wk=np.array([0,0,0,0,0], dtype=np.float64)):

        """
        Implimentation of Constant Turn Rate and Velocity Motion Model. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        Citation:
        Schubert, R., Richter, E., Wanielik, G.: Comparison and evaluation of advanced motion
        models for vehicle tracking. In: 2008 11th international conference on information fusion 2008,
        pp. 1-6. IEEE.

        https://ieeexplore.ieee.org/abstract/document/4632283

        """

        self.Wk = Wk

        return
    
    def update(self, mXkmve1=np.array([0,0,0,0,0], dtype=np.float64), sXkmve1=np.array([0,0,0,0,0], dtype=np.float64), dt=0.1):

        """Update function which is called when a new state estimate is received."""
        
        if mXkmve1[4] == 0:
            mXgXkmve1 = np.array([mXkmve1[0]+(mXkmve1[3]*np.cos(mXkmve1[2])*dt),
                                  mXkmve1[1]+(mXkmve1[3]*np.sin(mXkmve1[2])*dt),
                                  mXkmve1[2],
                                  mXkmve1[3],
                                  mXkmve1[4]])
            
            sXgXkmve1 = np.array([np.sqrt((sXkmve1[0]**2) + ((dt*np.cos(mXkmve1[2])*self.Wk[3])**2) + ((dt*mXkmve1[4]*np.sin(mXkmve1[2])*self.Wk[2])**2)),
                                  np.sqrt((sXkmve1[1]**2) + ((dt*np.sin(mXkmve1[2])*self.Wk[3])**2) + ((dt*mXkmve1[4]*np.cos(mXkmve1[2])*self.Wk[2])**2)),
                                  sXkmve1[2],
                                  sXkmve1[3],
                                  sXkmve1[4]])
        
        else:   
            mXgXkmve1 = np.array([mXkmve1[0] + (mXkmve1[3]/mXkmve1[4]) * (np.sin(mXkmve1[4] * dt  + mXkmve1[2]) - np.sin(mXkmve1[2])),
                                  mXkmve1[1] + (mXkmve1[3]/mXkmve1[4]) * (np.cos(mXkmve1[2]) - np.cos(mXkmve1[4] * dt + mXkmve1[2])),
                                  mXkmve1[2] + mXkmve1[4]*dt,
                                  mXkmve1[3],
                                  mXkmve1[4]])
            
            sXgXkmve1 = np.array([np.sqrt((sXkmve1[0]**2) + ((dt*np.cos(mXkmve1[2])*self.Wk[3])**2) + ((dt*mXkmve1[4]*np.sin(mXkmve1[2])*self.Wk[2])**2)),
                                  np.sqrt((sXkmve1[1]**2) + ((dt*np.sin(mXkmve1[2])*self.Wk[3])**2) + ((dt*mXkmve1[4]*np.cos(mXkmve1[2])*self.Wk[2])**2)),
                                  sXkmve1[2],
                                  sXkmve1[3],
                                  sXkmve1[4]])
            
        return mXgXkmve1, sXgXkmve1