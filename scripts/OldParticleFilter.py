#!/usr/bin/python3
import numpy as np
import rospy
import time
import uncertainties as unc
from math import pow, sqrt, sin, cos, asin, acos, atan2, inf, dist
import message_filters
from uncertainties import unumpy as unp
from std_msgs.msg import Float32MultiArray, Header
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped,PoseWithCovariance
from visualization_msgs.msg import Marker
import os

def resizeMatrix(matrix, size):

    newMatrix = np.zeros((size, size))

    newMatrix[:size/2, :size/2] = matrix

    return newMatrix


class IntentionModel:
        
    ########################################################
    #   INTENTION MODEL: u_t_k = KNOWN QUANTITY VIA CONNECTED VEHICLES
    #   CONTROL SPACE: u_t_k : [v_t_k, Φ_t_k]
    ########################################################

        def __init__(self):

            rospy.Subscriber('/target/control_space', Float32MultiArray, self.update_u_t_k)

            return

        def update_u_t_k(self, data):
            
            self.u_t_k = data

            rospy.loginfo(f"Observed Intention: v_t_k={round(self.u_k[0], 4)}, Φ_t_k={round(self.u_k[1], 4)}")

            return np.array(self.u_t_k)
        
    
class AckermanMotionModel:

    ########################################################
    #   MOTION MODEL: dot_x_t_k = f(x_t_kmve1, u_t_k, w_t_k) : NONLINEAR NON-GAUSSIAN
    #   CONTROL SPACE: u_t_k : [v_t_k, Φ_t_k]
    #   STATE SPACE: u_t_k : [x_t_k, y_t_k, θ_t_k, Φ_t_k]
    ########################################################

    def __init__(self):

        # Initialize Published ROS Messages
        self.state_pub = rospy.Publisher("/robot/particle_filter/motion_model/state_space", Float32MultiArray, queue_size=1)

        self.target_length = 1    

        return
    
    def getdot_x_t_k(self, x_t_kmve1, u_t_k, w_t_k):

        return np.array([[u_t_k[0]*np.cos(x_t_kmve1[2]) + w_t_k[0]], 
                         [u_t_k[0]*np.sin(x_t_kmve1[2]) + w_t_k[1]],
                         [(u_t_k[0]/self.target_length)*u_t_k[1]+ w_t_k[2]],
                         [u_t_k[1]]])
    
    def getw_t_k(self):

        return np.array([[np.random.normal(0, 0.01)], 
                         [np.random.normal(0, 0.01)], 
                         [np.random.normal(0, 0.005)]])
    
    def update(self, x_t_kmve1, u_t_kmve1, dk):                   

        return x_t_kmve1 + self.getdot_x_t_k(x_t_kmve1, u_t_kmve1, self.getw_t_k)*dk
    

class NonHolonomicMotionModel:

    ########################################################
    #   MOTION MODEL: dot_x_t_k = f(x_t_kmve1, u_t_k, w_t_k) : NONLINEAR NON-GAUSSIAN
    #   CONTROL SPACE: u_t_k : [v_t_k, Φ_t_k]
    #   STATE SPACE: u_t_k : [x_t_k, y_t_k, θ_t_k, Φ_t_k]
    ########################################################

    def __init__(self):

        # Initialize Published ROS Messages
        self.state_pub = rospy.Publisher("/robot/particle_filter/motion_model/state_space", Float32MultiArray, queue_size=1)

        self.target_length = 1    

        return
    
    def getdot_x_t_k(self, x_t_kmve1, u_t_k, w_t_k):

        return np.array([[u_t_k[0]*np.cos(x_t_kmve1[2]) + w_t_k[0]], 
                         [u_t_k[0]*np.sin(x_t_kmve1[2]) + w_t_k[1]],
                         [(u_t_k[0]/self.target_length)*u_t_k[1]+ w_t_k[2]],
                         [u_t_k[1]]])
    
    def getw_t_k(self):

        return np.array([[np.random.normal(0, 0.01)], 
                         [np.random.normal(0, 0.01)], 
                         [np.random.normal(0, 0.005)]])
    
    def update(self, x_t_kmve1, u_t_kmve1, dk):                   

        return x_t_kmve1 + self.getdot_x_t_k(x_t_kmve1, u_t_kmve1, self.getw_t_k)*dk
    

class SensorModel:

    ########################################################
    #   SENSOR MODEL: z_t_k = h(x_t_k, x_s_k, v_t_k)
    #   STATE SPACE: u_t_k : [x_t_k, y_t_k, θ_t_k, Φ_t_k]
    ########################################################

    def __init__(self):

        # Initialize Published ROS Messages
        self.state_pub = rospy.Publisher("/robot/particle_filter/sensor_model/predicted_state", PoseWithCovarianceStamped, queue_size=1)

        # Initialize Subscribed ROS Messages
        self.pointcloud = message_filters.Subscriber("/robot/lidar_points", PointCloud2)
        self.robot_pose = message_filters.Subscriber("/robot/amcl_pose", PoseWithCovarianceStamped)
        self.target_pose = message_filters.Subscriber("/target/amcl_pose", PoseWithCovarianceStamped)
        ts = message_filters.ApproximateTimeSynchronizer([self.pointcloud, self.robot_pose, self.target_pose], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.observationCB)

        # Initialize Observation
        self.observation = None

        # Initalize Start Message
        rospy.loginfo("Starting Sensor Model")


    def distance3d(self, x1,y1,z1,x2,y2,z2):

        # Return Magnitude of 3D Vector Representing Distance From Target To Observations
        return sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))
    
    
    def observationSpace(self, point):

        return np.array([sqrt((point[0]**2)+(point[1]**2)), atan2(point[1], point[0])])
    
    
    def observationCB(self, pointcloud, robot, target):

        try:
            
            self.fake_target_orientation = target.pose.pose.orientation
            self.fake_robot_orientation = robot.pose.pose.orientation

            shortest = inf

            for each in pc2.read_points(pointcloud, skip_nans=True, field_names = ("x", "y", "z")):

                distance = self.distance3d(target.pose.pose.position.x, target.pose.pose.position.y, target.pose.pose.position.z, each[0], each[1], each[2])

                if distance < shortest:

                    shortest = distance

                    point = each

            self.observation = self.observationSpace(point)

            self.update(self.observation)

            return

        except:

            rospy.logwarn("Unable to make observation")

            return
    
    
    def getCovarience(self, observation):

        # Taken From Ouster 1 Datasheet: https://data.ouster.io/downloads/datasheets/datasheet-gen1-v2p0-os1.pdf
        # https://uncertaintycalculator.com/

        if 0.8 < observation[0] < 1:   
            return np.array([[sqrt(((sin(observation[1])*0.01)**2)+((observation[0]*cos(observation[1])*np.deg2rad(0.01))**2)), 0],
                             [0, sqrt(((cos(observation[1])*0.01)**2)+((observation[0]*sin(observation[1])*np.deg2rad(0.01))**2))]])
        
        elif 1 < observation[0] < 20:   
            return np.array([[sqrt(((sin(observation[1])*0.011)**2)+((observation[0]*cos(observation[1])*np.deg2rad(0.01))**2)), 0],
                             [0, sqrt(((cos(observation[1])*0.011)**2)+((observation[0]*sin(observation[1])*np.deg2rad(0.01))**2))]])
        
        elif 20 < observation[0] < 50:   
            return np.array([[sqrt(((sin(observation[1])*0.03)**2)+((observation[0]*cos(observation[1])*np.deg2rad(0.01))**2)), 0],
                             [0, sqrt(((cos(observation[1])*0.03)**2)+((observation[0]*sin(observation[1])*np.deg2rad(0.01))**2))]])
              
        else:
            return np.array([[sqrt(((sin(observation[1])*0.05)**2)+((observation[0]*cos(observation[1])*np.deg2rad(0.01))**2)), 0],
                             [0, sqrt(((cos(observation[1])*0.05)**2)+((observation[0]*sin(observation[1])*np.deg2rad(0.01))**2))]])
        
        
    def getState(self, observation):

            return np.array([observation[0]*cos(observation[1]), observation[0]*sin(observation[1])])
    
    def publish(self, mean, covariance):

        theta = self.fake_target_orientation.z - self.fake_robot_orientation.z

        covarianceRotated = covariance*np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        visualizer = PoseWithCovarianceStamped()
        visualizer.pose.pose.position.x = mean[0]
        visualizer.pose.pose.position.y = mean[1]
        visualizer.pose.pose.orientation = self.fake_target_orientation
        visualizer.pose.covariance = np.zeros(36)
        visualizer.pose.covariance[0] = covarianceRotated[0][0]*((mean[0]**2)+(mean[1]**2))
        visualizer.pose.covariance[1] = covarianceRotated[0][1]*((mean[0]**2)+(mean[1]**2))
        visualizer.pose.covariance[6] = covarianceRotated[1][0]*((mean[0]**2)+(mean[1]**2))
        visualizer.pose.covariance[7] = covarianceRotated[1][1]*((mean[0]**2)+(mean[1]**2))
        visualizer.header.stamp = rospy.Time.now()
        visualizer.header.frame_id = "map"
        visualizer

        self.state_pub.publish(visualizer)

        return

    def update(self, observation):

        H_x_t = self.getState(observation)

        w_x_t = self.getCovarience(observation)

        self.publish(H_x_t, w_x_t)

        return H_x_t, w_x_t
    
    
class ParticleFilter:

    def __init__(self):

        # Initalize Parameters
        self.num_particles = 100
        self.x_range_particles = 10
        self.x_origin_particles = 0
        self.y_range_particles = 10
        self.y_origin_particles = 0
        
        # Initalize Particles
        self.particles = self.initialize_particles()

        return
    
    def initialize_particles(self):

        particles = []

        for each in range(self.num_particles):

            # Initialzie new particle.
            particle = Pose()

            # Initialize the particle's position and orientation randomly.
            particle.position.x = np.random.uniform(self.x_origin_particles - self.x_range_particles, self.x_origin_particles + self.x_range_particles)

            particle.position.y = np.random.uniform(self.y_origin_particles - self.y_range_particles, self.y_origin_particles + self.y_range_particles)

            particle.orientation.z = np.random.uniform(0, 2 * np.pi)

            particles.append(particle)

        return particles
        
    def prediction(self, x_kmve1, u_k, P_kmve1, dk):

        # Predict State Estimate
        x_kgkmve1 = MotionModel.update(x_kmve1, u_k, dk)

        # Predict State Covarience Estimate    
        P_kgkmve1 = (MotionModel.A_kmve1 @ P_kmve1 @ np.transpose(MotionModel.A_kmve1)) + self.Q_k
        
        return x_kgkmve1, P_kgkmve1
    
    def correction(self, z_k, x_kgkmve1, P_kgkmve1):

        # Calculate Near Optimal Kalman Gain
        s_kgk = (SensorModel.C_k @ P_kgkmve1 @ np.transpose(SensorModel.C_k)) + self.R_k
        K_k = (P_kgkmve1 @ np.transpose(SensorModel.C_k) @ np.linalg.pinv(s_kgk))

        # Calculate Mean in Correction
        x_k = x_kgkmve1 + K_k @ (z_k - SensorModel.update(x_kgkmve1))

        # Calculate Covarience in Correction
        P_k = (np.identity(3) - K_k @ SensorModel.C_k) @ P_kgkmve1
    
if __name__ == '__main__':

    rospy.init_node('particle_filter', anonymous=True)
    SensorModel = SensorModel()
    rate = rospy.Rate(10)
    #time.sleep(5)
    try:
        while not rospy.is_shutdown():
            rate.sleep()

    except rospy.ROSInterruptException:
        pass