#!/usr/bin/python3
import numpy as np
import rospy
from math import sqrt, atan2, inf, exp
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from vehicular_ws.src.filters.scripts.OldCTRVMotionModel import CTRVMotionModel
from AckermanMotionModel import AckermanMotionModel
from LidarTargetTracker import LidarTargetTracker
from vehicular_ws.src.filters.scripts.OldLidarSensorModel import LidarSensorModel
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import message_filters
import tf

class ParticleFilter(object):

    def __init__(self, N=1000, distribution='gaussian', visualize=True):

        # Initalize Parameters
        self.num_particles = 10

        self.distribution = distribution
        self.N = N

        # Initialize Motion Model
        self.MotionModel = CTRVMotionModel()

        # Initialize Intention Model
        self.Intention = AckermanMotionModel(length=1, visualize=True)

        # Initialize Sensor Reading
        self.Sensor = LidarTargetTracker(visualize=True)

        # Initialize Sensor Reading
        self.SensorModel = LidarSensorModel(visualize=True)

        self.visualize = visualize

        # Initialize Particles

        if self.visualize == True:
            self.particle_pub = rospy.Publisher("/robot/particlefilter/particles", PoseArray, queue_size=1)

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

        return particles
    
    def predict(self, particles, dt=0.1):

        MX_k, SX_k = self.MotionModel.update(particles, dt)

        particles[:] = np.random.normal(MX_k, SX_k)

        return particles

    def predict_particles(self, dt):

        updated_particles = []

        for particle in self.particles:

            updated_particles.append(self.MotionModel.update(particle, dt))

        self.particles = updated_particles

        return
    
    def weigh_particles(self, Z_k, W_k):

        weights = []
        
        for particle in self.particles:
            error = np.linalg.norm(Z_k - particle[:3])**2
            weight = exp((-0.5 * error))
            weights.append(weight)

        return weights
    
    def resample_particles(self, weights):

        probabilities = weights / np.sum(weights)
        index_numbers = np.random.choice(self.num_particles,size=self.num_particles,p=probabilities) 
        selected_particles = [self.particles[i] for i in index_numbers]
        self.particles = selected_particles

        return
    
    def apply_noise(self):
        noise= np.concatenate(
        (
            np.random.normal(0.0, 0.05, (self.num_particles,1)),
            np.random.normal(0.0, 0.05, (self.num_particles,1)),
            np.random.normal(0.0, 0.03, (self.num_particles,1)),
            np.random.normal(0.0, 0.01, (self.num_particles,1)),
            np.random.normal(0.0, 0.01, (self.num_particles,1))
        
        ),
        axis=1)
    
        self.particles += noise
        return
    
    def update(self, observation, dt=0.1):

        self.predict_particles(dt)

        Z_kmve1, W_kmve1 = self.SensorModel.update(observation)

        weights = self.weigh_particles(Z_kmve1, W_kmve1)

        self.resample_particles(weights)

        if self.visualize == True: 
            self.visualizer()

        return

    def visualizer(self):

        poseArray = PoseArray()

        poseArray.header.stamp = rospy.Time.now()

        poseArray.header.frame_id = "map"

        for particle in self.particles:

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
    
if __name__ == '__main__':



    ParticleFilter = ParticleFilter(visualize=True)
    rospy.init_node('particle_filter', anonymous=True)
    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():

            ParticleFilter.initialize_particles(
            
    except rospy.ROSInterruptException:
        pass