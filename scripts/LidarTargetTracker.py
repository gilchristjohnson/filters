#!/usr/bin/python3
import numpy as np
import rospy
from math import sqrt, sin, cos, atan2, inf
import message_filters
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf

class LidarTargetTracker(object):

    def __init__(self, visualize=False):

        """
        Implimentation of Simulated LiDAR Based Target Tracker. By Gilchrist Johnson (gilchristjohnson@virginia.edu), 2024.

        """

        self.visualize = visualize

        self.obseravtion_pub = rospy.Publisher("/robot/lidar_target_tracker/observation", Float32MultiArray, queue_size=1)
        self.pointcloud = message_filters.Subscriber("/robot/lidar_points", PointCloud2)
        self.robot_pose = message_filters.Subscriber("/robot/amcl_pose", PoseWithCovarianceStamped)
        self.target_pose = message_filters.Subscriber("/target/amcl_pose", PoseWithCovarianceStamped)
        ts = message_filters.ApproximateTimeSynchronizer([self.pointcloud, self.robot_pose, self.target_pose], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.observationCB)

        self.observation = None
        self.point = None

        if self.visualize == True:
            self.state_pub = rospy.Publisher("/robot/lidar_target_tracker/marker", Marker, queue_size=1)
        
        return

    def distance3d(self, x1,y1,z1,x2,y2,z2):

        return sqrt(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))
    
    def observationSpace(self, point, target):

        orientation = target.pose.pose.orientation

        orientation = [orientation.x, orientation.y, orientation.z, orientation.w]

        x, y, z = tf.transformations.euler_from_quaternion(orientation)

        return np.array([sqrt((point[0]**2)+(point[1]**2)), atan2(point[1], point[0]), z]) 
    
    def observationCB(self, pointcloud, robot, target):

        #try:

            shortest = inf

            for each in pc2.read_points(pointcloud, skip_nans=True, field_names = ("x", "y", "z")):

                distance = self.distance3d(target.pose.pose.position.x, target.pose.pose.position.y, target.pose.pose.position.z, each[0], each[1], each[2])

                if distance < shortest:

                    shortest = distance

                    self.point = each

            self.observation = self.observationSpace(self.point, target)

            msg = Float32MultiArray()

            msg.data = [self.observation[0], self.observation[1],self.observation[2]]

            self.obseravtion_pub.publish(msg)

            if self.visualize == True:
                self.visualizer(robot)

            #return

        #except:

            return      
        
    def visualizer(self, robot):

        visualizer = Marker()
        visualizer.header.frame_id = "robot/lidar_points"
        visualizer.header.stamp = rospy.Time.now()
        visualizer.ns = "laser"
        visualizer.id = 0
        visualizer.type = Marker.LINE_STRIP
        visualizer.action = Marker.ADD
        visualizer.scale.x = 0.01
        visualizer.pose.orientation.x = 0
        visualizer.pose.orientation.y = 0
        visualizer.pose.orientation.z = 0
        visualizer.pose.orientation.w = 1
        visualizer.pose.position.x = 0.0
        visualizer.pose.position.y = 0.0
        visualizer.pose.position.z = 0.0

        visualizer.points = []

        origin = Point()
        origin.x = robot.pose.pose.position.x
        origin.y = robot.pose.pose.position.y
        origin.z = -1.05
        visualizer.points.append(origin)

        center = Point()
        #center.x = self.point[0]
        #center.y = self.point[1]
        center.x = self.observation[0]*cos(self.observation[1])
        center.y = self.observation[0]*sin(self.observation[1])
        center.z = -1.05
        visualizer.points.append(center)

        visualizer.color.a = 1.0
        visualizer.color.r = 0.0
        visualizer.color.g = 1.0
        visualizer.color.b = 0.0

        self.state_pub.publish(visualizer)

        return
    
if __name__ == '__main__':

    rospy.init_node('lidar_target_tracker', anonymous=True)
    rate = rospy.Rate(10)

    LidarTargetTracker=LidarTargetTracker()

    try:
        while not rospy.is_shutdown():
            rospy.sleep

            
    except rospy.ROSInterruptException:
        pass