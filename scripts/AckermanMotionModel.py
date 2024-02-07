#!/usr/bin/python3
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from math import sin, cos, tan

class AckermanMotionModel(object):

    def __init__(self, U_k=np.array([[0],[0]]), X_k=np.array([[0],[0],[0]]), length=1, visualize=False):

        """
        Implimentation of Ackerman Motion Model. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        Citation:
        Modern Robotics:  Mechanics, Planning, and Control," by Kevin Lynch and Frank Park, Cambridge University Press 2017
        https://hades.mech.northwestern.edu/images/7/7f/MR.pdf

        """

        self.X_k = X_k

        self.U_k = U_k

        self.length = length

        self.visualize = visualize

        if self.visualize == True:
            self.state_pub = rospy.Publisher("/robot/ackerman_motion_model/estimated_state", PoseWithCovarianceStamped, queue_size=1)
        
        return
    
    def update(self, U_k, X_kmve1, dt):
        """Update function which is called when a new state estimate is received."""

        self.X_k = np.array([[X_kmve1[0] + U_k[0]*cos(X_kmve1[2])*dt],
                             [X_kmve1[1] + U_k[0]*sin(X_kmve1[2])*dt],
                             [X_kmve1[2] + (U_k[0]/self.length)*tan(U_k[1])*dt]])

        if self.visualize == True: 
            self.visualizer()
                 
        return self.X_k
    
    def visualizer(self):

        visualizer = PoseWithCovarianceStamped()

        visualizer.pose.pose.position.x = self.X_k[0]
        visualizer.pose.pose.position.y = self.X_k[1]
        visualizer.pose.pose.position.z = 0

        orientation = tf.transformations.quaternion_from_euler(0,0,self.X_k[2])
        visualizer.pose.pose.orientation.x = orientation[0]
        visualizer.pose.pose.orientation.y = orientation[1]
        visualizer.pose.pose.orientation.z = orientation[2]
        visualizer.pose.pose.orientation.w = orientation[3]

        visualizer.pose.covariance = np.zeros(36)

        visualizer.header.stamp = rospy.Time.now()
        visualizer.header.frame_id = "map"

        self.state_pub.publish(visualizer)

        return
