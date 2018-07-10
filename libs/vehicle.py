import math
class Vehicle:
    def __init__(self):
        self.LENGTH = 4.5  # [m]
        self.WIDTH = 2.0  # [m]
        self.BACKTOWHEEL = 1.0  # [m]
        self.WHEEL_LEN = 0.3  # [m]
        self.WHEEL_WIDTH = 0.2  # [m]
        self.TREAD = 0.7  # [m]
        self.WB = 2.5  # [m] wheel base

        self.MAX_STEER = math.radians(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = math.radians(30.0)  # maximum steering speed [rad/s]
        #self.MAX_DSTEER = []  # maximum steering speed [rad/s]
        self.MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
        self.MIN_SPEED = 0.0 / 3.6  # minimum speed [m/s]
        self.MAX_ACCEL = 1.0  # maximum accel [m/ss]
        self.MAX_BRAKE = 1.0  # maximum brake [m/ss]
        pass

class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0, v=0.0, steer = 0.0, brake = 0.0, throttle = 0.0, acc = 0.0, offroad = 0):
        self.x = x
        self.y = y
        self.z = z
        self.v = v
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.steer = steer
        self.brake = brake
        self.throttle = throttle
        self.acc = acc
        self.offroad = offroad
        self.errorDis = 0
        self.errorYaw = 0
        self.setV = 0
