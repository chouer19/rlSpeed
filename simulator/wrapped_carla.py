from __future__ import print_function

import argparse
import logging
import random
import time
import math

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

import sys
sys.path.append('controller/')
from controller import *


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
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._control = VehicleControl()
        self._control.throttle = 0.58
        self._startID = [0,2,4,6,11,13,17,19,21,24,30,39,55,57,66,70]
        self.state = State()

    def _initialize_game(self):
        self._on_new_episode()

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        player_start = self._startID[random.randint(0,15)]
        #ID = random.choice([3,4,5,6,8,12,13,15,16])
        #player_start = self._startID[3]
        #player_start = self._startID[4]
        #player_start = self._startID[5]
        #player_start = self._startID[6]
        #player_start = self._startID[8]
        #player_start = self._startID[12]
        #player_start = self._startID[13]
        #player_start = self._startID[15]
        #player_start = self._startID[16]
        player_start = 0
        print('Starting new episode...',player_start)
        self.client.start_episode(player_start)

    def initialize_game(self):
        self._initialize_game()

    def new_game(self):
        self._on_new_episode()

    def frame_step(self, ste, thro, bra):
        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraRGB', None)

        transform = measurements.player_measurements.transform
        x,y,z = transform.location.x, transform.location.y, transform.location.z
        roll,pitch,yaw = transform.rotation.roll, transform.rotation.pitch, math.radians(transform.rotation.yaw)
        acceleration = measurements.player_measurements.acceleration
        forward_speed = measurements.player_measurements.forward_speed
        steer, brake, throttle = self._control.steer, self._control.brake, self._control.throttle
        intersection_offroad = measurements.player_measurements.intersection_offroad

        self.state = State(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, v=forward_speed, steer = self._control.steer, brake = self._control.brake, throttle = self._control.throttle, acc = acceleration, offroad = intersection_offroad)

        self._control.steer = ste
        self._control.brake = bra
        self._control.throttle = thro
        self.client.send_control(self._control)
        return self.state
