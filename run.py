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
import wrapped_carla as simulator
sys.path.append('dqn/')
import dqn

from carla import image_converter
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.planner.map import CarlaMap
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

ACTIONS = 15
FRAME_PER_ACTION = 4
BATCH = 32

def Run(args , cont, s, readout, h_fc1):
    with make_carla_client(args.host, args.port) as client:
        #init game
        game = simulator.CarlaGame(client, args)
        game.initialize_game()
        #game.new_game()

        # define the cost function
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

        # do nothing
        throttle, brake, steer = 0,0,0
        action_value = 0
        throttle = max(0, action_value)
        brake = max(0, -1 * action_value)
        action_array = value_to_array(action_value)
        con.state = game.frame_step(steer,throttle,brake)
        terminal, reward, image = render.new_frame(con.state.v)

        x_t = cv2.cvtColor(cv2.resize(image, (160, 160)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t), axis=2)

        Replay = deque()

        # saving and loading networks
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        timer = Timer()
        epsilon = INITIAL_EPSILON
        t = 0
        while True:
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            readout_t = readout.eval(feed_dict={s : [s_t]})[0]

            # get an action
            if t % FRAME_PER_ACTION == 0:
                if random.random() < epsilon:
                    action_value = round(random.random() * 2 - 1,2)
                    action_value = max(-0.95, action_value)
                    action_value = min(0.95, action_value)
                    action_array = value_to_array(action_value)
                else:
                    action_array = readout_t
                    action_value = array_to_value(action_array)

            # do an action and get speed, acc and so on
            #throttle, brake, steer= con.pp_control()
            #throttle, brake, steer= con.pp_control()
            _, _, steer= con.stanely_control()
            throttle = max(0, action_value)
            brake = max(0, -1 * action_value)
            con.state = game.frame_step(steer,throttle,brake)
            # render and get reward, terminal, img
            terminal, reward, image = render.new_frame(con.state.v)
            
            if t % FRAME_PER_ACTION == 0:
                x_t1 = cv2.cvtColor(cv2.resize(image, (160, 160)), cv2.COLOR_BGR2GRAY)
                ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
                x_t1 = np.reshape(x_t1, (160, 160, 1))
                s_t1 = np.append(x_t1, s_t[:, :, :9], axis=2)
                # store the transition in Reply
                Repley.append((s_t, a_t, r_t, s_t1, terminal))
                if len(Repley) > REPLAY_MEMORY:
                    Repley.popleft()
            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA *
                                       ( np.max(readout_j1_batch[i][0:3] ) + \
                                         np.max(readout_j1_batch[i][3:5] ) + \
                                         np.max(readout_j1_batch[i][5:7] ) + \
                                         np.max(readout_j1_batch[i][7:9] ) + \
                                         np.max(readout_j1_batch[i][9:11] ) + \
                                         np.max(readout_j1_batch[i][11:13]) +\
                                         np.max(readout_j1_batch[i][13:15])
                                       )
                                      )
                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                )
            # update the old values
            s_t = s_t1
            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, 'saved_networks/rlSpeed-dqn', global_step = t)

            t += 1
        pass

def value_to_array(action_value):
    action_array = np.zeros(ACTIONS)
    action_value = int(action_balue) * 100
    action_value = min(95, action_value)
    action_value = max(-95, action_value)
    if action_value >=0:
        action_array[-1] = 1
    else:
        action_array[-2] = 1
        action_value = abs(action_value)
    action_array[int(action_value / 32)] = 1
    action_value = action_value % 32
    action_array[int(action_value/16)+3] = 1
    action_value = action_value % 16
    action_array[int(action_value/8)+5] = 1
    action_value = action_value % 8
    action_array[int(action_value/4)+7] = 1
    action_value = action_value % 4
    action_array[int(action_value/2)+9] = 1
    action_value = action_value % 2
    action_array[int(action_value/1)+11] = 1
    return action_array
    pass

def array_to_value(action_array):
    action_value = np.argmax(action_array[0:3]) * 32 +\
                   np.argmax(action_array[3:5]) * 16 + \
                   np.argmax(action_array[5:7]) * 8 + \
                   np.argmax(action_array[7:9]) * 4 + \
                   np.argmax(action_array[9:11]) * 2 + \
                   np.argmax(action_array[11:13]) * 1
    action_value = action_value * (np.argmax(action_array[13:15]) * 2 -1)
    return action_value
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
