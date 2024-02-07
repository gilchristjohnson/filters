#!/usr/bin/python3
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as rotation
from math import sin, cos, sqrt

class LidarSensorModel(object):

    def __init__(self, Vk=np.array([0.05, np.deg2rad(0.01), 0.003]), visualize=False):

        """
        Implimentation of Simulated LiDAR Based Sensor Model. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        """

        self.Vk = Vk

        self.visualize = visualize

        if self.visualize == True:
            self.state_pub = rospy.Publisher("/robot/lidar_sensor_model/estimated_state", PoseWithCovarianceStamped, queue_size=1)
        
        return
    
    def update(self, observation):
        """Update function which is called when a new state estimate is received."""

        Zk = np.array([observation[0]*cos(observation[1]),
                       observation[0]*sin(observation[1]),
                       observation[2]])

        Wk = np.array([sqrt((cos(observation[1]) * self.Vk[0])**2 + (observation[0] * sin(observation[1]) * self.Vk[1])**2),
                       sqrt((sin(observation[1]) * self.Vk[0])**2 + (observation[0] * cos(observation[1]) * self.Vk[1])**2),
                       self.Vk[1]**2])       

        if self.visualize == True: 
            self.visualizer(Zk, Wk)
                 
        return Zk, Wk
    
    def visualizer(self, Zk, Wk):

        visualizer = PoseWithCovarianceStamped()
        
        visualizer.pose.pose.position.x = Zk[0]
        visualizer.pose.pose.position.y = Zk[1]
        visualizer.pose.pose.position.z = 0

        #R = np.array([[cos(Zk[2]), -sin(Zk[2])], [sin(Zk[2]), cos(Zk[2])]])

        #S = np.array([[Wk.flatten()[0], 0], [0, Wk.flatten()[1]]])

        #T = S*R

        orientation = tf.transformations.quaternion_from_euler(0,0,Zk[2])
        visualizer.pose.pose.orientation.x = orientation[0]
        visualizer.pose.pose.orientation.y = orientation[1]
        visualizer.pose.pose.orientation.z = orientation[2]
        visualizer.pose.pose.orientation.w = orientation[3]

        visualizer.pose.covariance = np.zeros(36)

        visualizer.header.stamp = rospy.Time.now()
        visualizer.header.frame_id = "map"

        self.state_pub.publish(visualizer)

        return