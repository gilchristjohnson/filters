#!/usr/bin/python3
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sin, cos, sqrt

class CTRVMotionModel(object):

    def __init__(self, Vk=np.array([0,0,0,0,0]), visualize=False):

        """
        Implimentation of Constant Turn Rate and Velocity Motion Model. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        Citation:
        Schubert, R., Richter, E., Wanielik, G.: Comparison and evaluation of advanced motion
        models for vehicle tracking. In: 2008 11th international conference on information fusion 2008,
        pp. 1-6. IEEE.

        https://ieeexplore.ieee.org/abstract/document/4632283

        """

        self.Vk = Vk

        self.visualize = visualize

        if self.visualize == True:
            self.state_pub = rospy.Publisher("/robot/ctrv_motion_model/estimated_state", PoseWithCovarianceStamped, queue_size=1)
        
        return
    
    def update(self, mX_kmve1=np.array([0,0,0,0,0]), sX_kmve1=np.array([0,0,0,0,0]), dt=0.1):
        """Update function which is called when a new state estimate is received."""

        if mX_kmve1[4] == 0:

            mXgXkmve1 = np.array([mX_kmve1[0] + (mX_kmve1[3] * cos(mX_kmve1[2]) * dt),
                                  mX_kmve1[1] + (mX_kmve1[3] * sin(mX_kmve1[2]) * dt),
                                  mX_kmve1[2],
                                  mX_kmve1[3],
                                  mX_kmve1[4]])
            
            sXgXkmve1 = np.array([sqrt((sX_kmve1[0]**2) + ((dt*cos(mX_kmve1[2])*self.Vk[3])**2) + ((dt*mX_kmve1[4]*sin(mX_kmve1[2])*self.Vk[2])**2)),
                                  sqrt((sX_kmve1[1]**2) + ((dt*sin(mX_kmve1[2])*self.Vk[3])**2) + ((dt*mX_kmve1[4]*cos(mX_kmve1[2])*self.Vk[2])**2)),
                                  sX_kmve1[2],
                                  sX_kmve1[3],
                                  sX_kmve1[4]])
            
        else:

            mXgXkmve1 = np.array([mX_kmve1[0] + (mX_kmve1[3]/mX_kmve1[4]) * (sin(mX_kmve1[4] * dt  + mX_kmve1[2]) - sin(mX_kmve1[2])),
                                  mX_kmve1[1] + (mX_kmve1[3]/mX_kmve1[4]) * (cos(mX_kmve1[2]) - cos(mX_kmve1[4] * dt + mX_kmve1[2])),
                                  mX_kmve1[2] + mX_kmve1[4]*dt,
                                  mX_kmve1[3],
                                  mX_kmve1[4]])   

            sXgXkmve1 = np.array([sqrt((sX_kmve1[0]**2) + ((dt*cos(mX_kmve1[2])*self.Vk[3])**2) + ((dt*mX_kmve1[4]*sin(mX_kmve1[2])*self.Vk[2])**2)),
                                  sqrt((sX_kmve1[1]**2) + ((dt*sin(mX_kmve1[2])*self.Vk[3])**2) + ((dt*mX_kmve1[4]*cos(mX_kmve1[2])*self.Vk[2])**2)),
                                  sX_kmve1[2],
                                  sX_kmve1[3],
                                  sX_kmve1[4]])     

        #if self.visualize == True: 
            #self.visualizer(mXgXkmve1, sXgXkmve1)
                 
        return mXgXkmve1, sX_kmve1

    def visualizer(self, mXk, sXk):

        visualizer = PoseWithCovarianceStamped()
        
        visualizer.pose.pose.position.x = mXk[0]
        visualizer.pose.pose.position.y = mXk[1]
        visualizer.pose.pose.position.z = 0

        orientation = tf.transformations.quaternion_from_euler(0,0,sXk[2])
        visualizer.pose.pose.orientation.x = orientation[0]
        visualizer.pose.pose.orientation.y = orientation[1]
        visualizer.pose.pose.orientation.z = orientation[2]
        visualizer.pose.pose.orientation.w = orientation[3]

        visualizer.pose.covariance = np.zeros(36)
        visualizer.pose.covariance

        visualizer.header.stamp = rospy.Time.now()
        visualizer.header.frame_id = "map"

        self.state_pub.publish(visualizer)

        return
