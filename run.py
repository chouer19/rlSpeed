#!/usr/bin/env python
from __future__ import print_function

import argparse
import logging
import random
import time
import math

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import tensorflow as tf
import cv2
import sys
sys.path.append("controller/")
from controller import *
from timer import *
import random
import numpy as np
import threading
from collections import deque
import time

sys.path.append('simulator/')
#from simulator import *
import wrapped_carla_sim as simulator
sys.path.append('dqn/')
import dqn

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

def Run(args , cont, s, readout, h_fc1):
    with make_carla_client(args.host, args.port) as client:
        #init game
        game = simulator.CarlaGame(client, args)

        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # do nothing
        con.state = game.frame_step(0,0,0)
        terminal, reward, image = render.new_frame(speed)

        Replay = deque()

        game.initialize_game()
        #game.new_game()

        timer = Timer()
        throttle, brake, steer = 0,0,0
        t = 0
        while True:
                #throttle, brake, steer= con.pp_control()
                #throttle, brake, steer= con.pp_control()
                throttle, brake, steer= con.stanely_control()
                
                print('setV realV acc brake')
                print(round(con.state.setV,2), round(con.state.v,2), round(throttle,2), round(brake,2) )
                print('cmd -> throttle, brake, steer')
                print(round(throttle,2), round(brake,2), round(steer,2))
                print('\n')
                con.state = game.frame_step(steer, throttle, brake)
                #con.state = game.frame_step(0, throttle, brake)
                #con.render()
                timer.lap()
                if False:
                    game.new_game()
        pass

def playGame(args):
    contr = Controller(filename = 'log/all.road')
    # network
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    Run(args , contr, s, readout, h_fc1)

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-r', '--road',
        metavar='R',
        default='waypoints',
        help='road location of waypoints road')
    argparser.add_argument(
        '-rl', '--road_length',
        metavar='RL',
        default=15,
        type=int,
        help='length of stright roads')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map_name',
        metavar='M',
        default='Town01',
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    playGame(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
