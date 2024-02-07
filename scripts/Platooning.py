#!/usr/bin/python3
import numpy as np
from filterpy.monte_carlo import systematic_resample
import rospy
from math import sqrt, atan2, inf, exp
from std_msgs.msg import Float32MultiArray
from ParticleFilter import ParticleFilter

if __name__ == '__main__':

    initial_parameters = [(0, 10), (0, 10), (0, np.pi/2), (0, 0), (0, 0)]
    rospy.init_node('platooning', anonymous=True)
    rate = rospy.Rate(10)
    ParticleFilter = ParticleFilter(parameters=initial_parameters, visualize=True)


    try:
        while not rospy.is_shutdown():
            rospy.sleep

            
    except rospy.ROSInterruptException:
        pass