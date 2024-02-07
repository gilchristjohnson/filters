#!/usr/bin/python3
import numpy as np
import time
import rospy
from math import sqrt, atan2, inf, exp
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from vehicular_ws.src.filters.scripts.OldCTRVMotionModel import CTRVMotionModel
from AckermanMotionModel import AckermanMotionModel
from LidarTargetTracker import LidarTargetTracker
from vehicular_ws.src.filters.scripts.OldLidarSensorModel import LidarSensorModel
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose
import tf

class ParticleFilter(object):

    def __init__(self, parameters, N=100, distribution='uniform', visualize=False):

        self.visualize = visualize

        self.MotionModel = CTRVMotionModel(Vk=np.array([0.05,0.05,0.003,0.005,0.0003]))

        #self.Intention = AckermanMotionModel(length=1, visualize=True)

        self.SensorModel = LidarSensorModel(Vk=np.array([0.05, np.deg2rad(0.01), 0.003]))

        self.state_pub = rospy.Publisher("/robot/particle_filter/estimated_state", PoseWithCovarianceStamped, queue_size=1)
        
        if self.visualize == True:
            self.particle_pub = rospy.Publisher("/robot/particle_filter/particles", MarkerArray, queue_size=1)

        self.particles, self.weights = self.initialize_particles(parameters, distribution, N)

        self.previous_time = rospy.get_time() 

        return
    
    def initialize_particles(self, parameters, distribution, N):

        particles = np.empty((N, len(parameters)))

        for index in range(len(parameters)):

            if distribution == 'uniform':

                particles[:, index] = np.random.uniform(parameters[index][0], parameters[index][1], size=N)

            elif distribution == 'gaussian':

                particles[:, index] = np.random.normal(parameters[index][0], parameters[index][1], size=N)

            else:

                raise ValueError(self.distribution + "is not a valid distribution type.")

        weights = np.ones(N)/N

        return particles, weights
    
    def prediction(self, particles, dt):

        for index in range(5):

            mXkgkmve1, sXkgkmve1 = self.MotionModel.update(mX_kmve1=particles[index], sX_kmve1=np.array([0, 0, 0, 0, 0]), dt=dt)

            particles[index] = np.random.multivariate_normal(mean=mXkgkmve1, cov=np.diag(sXkgkmve1))

        return particles
    
    def correction(self, particles, weights, observation):

        mX_k, sX_k = self.SensorModel.update(observation)

        distance = np.linalg.norm(particles[:, 0:2] - mX_k[0:2], axis=1)

        weights = (np.max(distance) - distance)**8

        return particles, weights
    
    def neff(self, weights):
        return 1. / np.sum(np.square(weights))
    
    def resample(self, particles, weights):

        probabilities = weights / np.sum(weights)
        index_numbers = np.random.choice(len(particles), size=len(particles),p=probabilities)

        particles = particles[index_numbers, :]
        weights = weights[index_numbers]
    
        return particles, weights
    
    def estimate(self, particles, weights):
        """returns mean and variance of the weighted particles"""

        pos = particles[:, 0:3]

        mean = np.average(pos, weights=weights, axis=0)

        var  = np.average((pos - mean)**2, weights=weights, axis=0)

        return mean, var
    
    def update(self, observation):

        dt = rospy.get_time() - self.previous_time

        self.previous_time = rospy.get_time()

        self.particles = self.prediction(self.particles, dt)

        self.particles, self.weights = self.correction(self.particles, self.weights, observation)

        if self.neff(self.weights) < len(self.particles)/2:
           
            self.particles, self.weights = self.resample(self.particles, self.weights)

        mean, var = self.estimate(self.particles, self.weights)

        print(mean, var)

        #self.visualizer(self.particles, self.weights)

        return
    
    def oldvisualizer(self, mXk, sXk):

        visualizer = PoseWithCovarianceStamped()
        
        visualizer.pose.pose.position.x = mXk[0]
        visualizer.pose.pose.position.y = mXk[1]
        visualizer.pose.pose.position.z = 0

        orientation = tf.transformations.quaternion_from_euler(0,0,mXk[2])
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

    def visualizer(self, particles, weights):

        marker_array = MarkerArray()

        for index in range(len(particles)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = particles[index][0]
            marker.pose.position.y = particles[index][1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = weights[index]
            marker.scale.y = weights[index]
            marker.scale.z = weights[index]
            marker.color.a = 1.0
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            marker_array.markers.append(marker)

        self.particle_pub.publish(marker_array)
    
"""
    
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

    def particlevisualizer(self, particles):

        poseArray = PoseArray()

        poseArray.header.stamp = rospy.Time.now()

        poseArray.header.frame_id = "map"

        for particle in particles:

            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            pose.position.z = 0

            orientation = tf.transformations.quaternion_from_euler(0,0,particle[2])
            pose.orientation.x = orientation[0]
            pose.orientation.y = orientation[1]
            pose.orientation.z = orientation[2]
            pose.orientation.w = orientation[3]

            poseArray.poses.append(pose)
            
        self.particle_pub.publish(poseArray)

        return

        
"""

if __name__ == '__main__':

    initial_parameters = [(0, 10), (-5, 5), (0, np.pi/2), (0, 0), (0, 0)]
    rospy.init_node('platooning', anonymous=True)
    rate = rospy.Rate(10)
    ParticleFilter = ParticleFilter(parameters=initial_parameters, visualize=True)

    rospy.Subscriber("/robot/lidar_target_tracker/observation", Float32MultiArray, ParticleFilter.update)

    try:
        while not rospy.is_shutdown():
            rospy.sleep

            
    except rospy.ROSInterruptException:
        pass
    

