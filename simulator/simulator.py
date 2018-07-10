import argparse
import logging
import random
import time
import math

import sys
sys.path.append(../libs/)
from vehicle import *

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line


def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need."""
    settings = CarlaSettings()
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        SeedVehicles = '00000',
        WeatherId=1,
        QualityLevel=args.quality_level)
    settings.randomize_seeds()
    return settings


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client    # build a client
        self._carla_settings = make_carla_settings(args)
        self._measurements = None
        self._control = VehicleControl()
        self._startID = [0,2,4,6,11,13,17,19,21,24,30,39,55,57,66,70]

    def _on_new_episode(self):
        self.client.load_settings(self._carla_settings)
        player_start = self._startID[random.randint(0,15)]
        print('Starting new episode...',player_start)
        self.client.start_episode(player_start)

    def new_game(self):
        self._on_new_episode()

    def get_measurements(self):
        measurements, _ = self.client.read_data()
        self._measurements = measurements
        return measurements

    def get_imu(self):
        measurements, _ = self.client.read_data()
        transform = measurements.player_measurements.transform
        acceleration = measurements.player_measurements.acceleration
        forward_speed = measurements.player_measurements.forward_speed
        intersection_offroad = measurements.player_measurements.intersection_offroad
        return transform, acceleration, forward_speed, intersection_offroad

    def get_state(self):
        measurements, _ = self.client.read_data()
        transform = measurements.player_measurements.transform
        x,y,z = transform.location.x, transform.location.y, transform.location.z
        roll,pitch,yaw = transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw
        acceleration = measurements.player_measurements.acceleration
        forward_speed = measurements.player_measurements.forward_speed
        steer, brake, throttle = self._control.steer, self._control.brake, self._control.throttle
        intersection_offroad = measurements.player_measurements.intersection_offroad
        return State(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, v=forward_speed, steer = steer, brake = brake, throttle = throttle, acc = acceleration, offroad = intersection_offroad)

    def get_control(self):
        return self._control

    def send_control(self, steer = 0, throttle = 0, brake = 0, reverse = False, hand_brake = False):
        self._control.steer = steer
        self._control.throttle = throttle
        self._control.brake = brake
        self._control.reverse = reverse
        self._control.hand_brake = hand_brake
        self.client.send_control(self._control)

    def get_states(self):
        return self._speed,self._error_dis,self._error_yaw, self._control.steer, self._offroad
