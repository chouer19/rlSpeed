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

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 320

ROAD_CENTER = 0.3

ROAD_WIDTH = 8.6
PIXEL_DENSITY = 10
FPS = 20

CAR_X = 10
CAR_Y = int(WINDOW_HEIGHT/2/PIXEL_DENSITY)

CAR_WIDTH = 2.4
CAR_LENGTH = 5.5
CAR_DUI = math.sqrt(CAR_WIDTH **2 + CAR_LENGTH ** 2)
#CAR_RECT =  pygame.Rect( ( int(WINDOW_WIDTH/4 - CAR_LENGTH/2 * PIXEL_DENSITY),  \
#                           int(WINDOW_HEIGHT/2- CAR_WIDTH/2 * PIXEL_DENSITY)), \
#                         ( int(CAR_LENGTH * PIXEL_DENSITY), int(CAR_WIDTH * PIXEL_DENSITY)))
CAR_RECT =  pygame.Rect( ( int(WINDOW_WIDTH/4 - 2 * PIXEL_DENSITY),  \
                           int(WINDOW_HEIGHT/2- CAR_WIDTH/2 * PIXEL_DENSITY)), \
                         ( int(CAR_LENGTH * PIXEL_DENSITY), int(CAR_WIDTH * PIXEL_DENSITY)))


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


class Timer(object):
    def __init__(self):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()

    def tick(self):
        self.step += 1

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


class CarlaGame(object):
    def __init__(self, carla_client, args):
        self.client = carla_client
        self._carla_settings = make_carla_settings(args)
        self._display = None
        self._enable_autopilot = args.autopilot
        self._is_on_reverse = False
        self._city_name = args.map_name
        self._map = CarlaMap(self._city_name, 16.43, 50.0) if self._city_name is not None else None
        self._map_shape = self._map.map_image.shape if self._city_name is not None else None
        #self._map_view = self._map.get_map(WINDOW_HEIGHT) if self._city_name is not None else None
        self._map_view = None
        self._position = None
        self._control = VehicleControl()
        self._control.throttle = 0.58
        self._location = [0,0,0]
        self._roll = 0.0
        self._started = False
        self._startID = [0,2,4,6,11,13,17,19,21,24,30,39,55,57,66,70]
        self._speed = 0
        self._offroad = 0
        self._terminal = False
        self._error_dis = 0
        self._error_yaw = 0
        self._road_left,self._road,self._road_right = self._load_road('waypoints',15)
        self.state = State()

    def _load_road(self,folder,length):
        points = []
        points_left = []
        points_right = []

        def readfile(name):
            with open(name) as f:
                line = f.readline()
                while line:
                    contents = line.split(',')
                    if len(contents) == 3:
                        x,y,yaw = float(contents[0]),  float(contents[1]), float(contents[2])
                        points.append(( x,y,yaw ))
                        points_left.append(( x + ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y - ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                        points_right.append(( x - ROAD_WIDTH/2 * math.sin(math.radians(yaw)), y + ROAD_WIDTH/2 * math.cos(math.radians(yaw)), yaw ))
                        pass
                    line = f.readline()

        n = folder + '/1.txt'
        readfile(n)
        for i in range(1,length):
            #load route i->j
            n = folder + '/' + str(i) + '-' + str(i+1) + '.txt'
            readfile(n)
            #load route i+1
            n = folder + '/' + str(i+1) + '.txt'
            readfile(n)

        return points_left, points, points_right


    def execute(self):
        """Launch the PyGame."""
        pygame.init()
        self._initialize_game()
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                self._on_loop()
                self._on_render()
        finally:
            pygame.quit()

    def _initialize_game(self):
        #if self._city_name is not None:
        if self._map_view is not None:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH + int((WINDOW_HEIGHT/float(self._map.map_image.shape[0]))*self._map.map_image.shape[1]), WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self._display = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

        logging.debug('pygame started')
        self._on_new_episode()

    def _on_new_episode(self):
        scene = self.client.load_settings(self._carla_settings)
        player_start = self._startID[random.randint(0,15)]
        print('Starting new episode...',player_start)
        self.client.start_episode(player_start)
        self._is_on_reverse = False

    def initialize_game(self):
        pygame.init()
        self._initialize_game()

    def new_game(self):
        self._on_new_episode()

    def frame_step(self):
        self._on_loop()
        return self._on_render()

    def set_steer(self,value):
        self._control.steer = value
        #self._control.steer += value
        self._control.steer = max(-1,self._control.steer)
        self._control.steer = min(1,self._control.steer)
    def get_states(self):
        return self._speed,self._error_dis,self._error_yaw, self._control.steer, self._offroad, self._roll

    def _on_loop(self):

        measurements, sensor_data = self.client.read_data()

        self._main_image = sensor_data.get('CameraRGB', None)

        # Print measurements every second.
        if True:
            if True:
                self._location = [
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.rotation.yaw]
                self._speed = measurements.player_measurements.forward_speed
                self._offroad = measurements.player_measurements.intersection_offroad
                self._offroad %= 1
                self._roll -= measurements.player_measurements.transform.rotation.roll
                

                #self._print_player_states()

            if self._map_view is not None:
                self._position = self._map.convert_to_pixel([
                    measurements.player_measurements.transform.location.x,
                    measurements.player_measurements.transform.location.y,
                    measurements.player_measurements.transform.location.z])

        control = self._get_keyboard_control(pygame.key.get_pressed())

        transform = measurements.player_measurements.transform
        x,y,z = transform.location.x, transform.location.y, transform.location.z
        roll,pitch,yaw = transform.rotation.roll, transform.rotation.pitch, math.radians(transform.rotation.yaw)
        acceleration = measurements.player_measurements.acceleration
        forward_speed = measurements.player_measurements.forward_speed
        steer, brake, throttle = self._control.steer, self._control.brake, self._control.throttle
        intersection_offroad = measurements.player_measurements.intersection_offroad

        self.state = State(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll, v=forward_speed, steer = self._control.steer, brake = self._control.brake, throttle = self._control.throttle, acc = acceleration, offroad = intersection_offroad)

    def _get_keyboard_control(self, keys):
        """
        Return a VehicleControl message based on the pressed keys. Return None
        if a new episode was requested.
        """
        if keys[K_r]:
            return None
        control = VehicleControl()
        if keys[K_LEFT] or keys[K_a]:
            self._control.steer += -0.02
            self._control.steer = max(self._control.steer,-1.0)
            control.steer = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            self._control.steer += 0.02
            self._control.steer = min(self._control.steer,1.0)
            control.steer = 1.0
        if keys[K_l] or  keys[K_SPACE]:
            self._control.steer = 0
        if keys[K_UP] or keys[K_w]:
            control.throttle = 1.0
            self._control.throttle = 0.6
            self._control.throttle = min(self._control.throttle,1)
            self._control.brake = 0.0
        if keys[K_DOWN] or keys[K_s]:
            control.brake = 1.0
            self._control.brake = 1.0
            self._control.throttle = 0.0
        if keys[K_q]:
            self._is_on_reverse = not self._is_on_reverse
        if keys[K_p]:
            self._enable_autopilot = not self._enable_autopilot
        control.reverse = self._is_on_reverse
        self._control.reverse = self._is_on_reverse
        return control

    def _print_player_states(self):
        message = 'location:({x:.1f},{y:.1f},{z:.1f})\tspeed:{speed:.1f} m/s\tsteer:{steer:.3f}\toffroad:{offroad:.0f} %\tmax distance:{dis:.2f} m\tdiff yaw:{yaw:.2f}'
        message = message.format(
            speed=self._speed,
            steer=self._control.steer,
            offroad=100 * self._offroad,
            dis=self._error_dis,
            x=self._location[0],
            y=self._location[1],
            z=self._location[2],
            yaw=self._error_yaw)
        print_over_same_line(message)

    def _on_render(self):
        if self._map_view is not None:
            array = self._map_view
            array = array[:, :, :3]

            new_window_width = \
                (float(WINDOW_HEIGHT) / float(self._map_shape[0])) * \
                float(self._map_shape[1])
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

            w_pos = int(self._position[0]*(float(WINDOW_HEIGHT)/float(self._map_shape[0])))
            h_pos = int(self._position[1] *(new_window_width/float(self._map_shape[1])))

            pygame.draw.circle(surface, [255, 0, 0, 255], (w_pos, h_pos), 6, 0)
            self._display.blit(surface, (WINDOW_WIDTH, 0))

        return self._draw_road()

    def _draw_road(self):
        terminal = False
        map_points = []
        map_points_left = []
        map_points_right = []
        outer_points = []
        mark = 0
        dis = 0
        nearDis = 999
        for i,road_point in enumerate(self._road):
            dis = math.sqrt((road_point[0] - self._location[0])**2 + (road_point[1] - self._location[1]) ** 2)
            if dis < nearDis:
                nearDis = dis
                mark = i
        #find forward for distance of 60 meters
        forward = 0 
        dis = 0 
        while dis < 75: 
            forward += 1
            dis = math.sqrt( (self._road[(mark + forward) % len(self._road)][0] - self._road[mark][0]) ** 2 + \
                             (self._road[(mark + forward) % len(self._road)][1] - self._road[mark][1]) ** 2 ) 
        #find backward for distance of 15 meters
        backward = 0 
        dis = 0 
        while dis < 20: 
            backward += 1
            dis = math.sqrt( (self._road[(mark - backward) % len(self._road)][0] - self._road[mark][0]) ** 2 + \
                             (self._road[(mark - backward) % len(self._road)][1] - self._road[mark][1]) ** 2 ) 
        #get center_points on image
        for i in range(mark-backward, mark + forward):
            #center_points
            Xct,Yct = self._road[i%len(self._road)][0] - self._location[0] , self._road[i%len(self._road)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            map_points.append(( int(X), int(Y) ))
            #left_points
            Xct,Yct = self._road_left[i%len(self._road_left)][0] - self._location[0] , self._road_left[i%len(self._road_left)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            outer_points.append(( int(X), int(Y) ))
            if CAR_RECT.collidepoint( int(X),int(Y) ):
                terminal = True
            #right_points
            Xct,Yct = self._road_right[i%len(self._road_right)][0] - self._location[0] , self._road_right[i%len(self._road_right)][1] - self._location[1]
            Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                        Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
            X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
            outer_points = [(int(X),int(Y))] + outer_points
            if CAR_RECT.collidepoint( int(X),int(Y) ):
                terminal = True

        pygame.draw.rect(self._display,(255,255,255), pygame.Rect((0,0),(WINDOW_WIDTH,WINDOW_HEIGHT)))
        pygame.draw.polygon(self._display, (0,0,0), outer_points, 0 )
        pygame.draw.lines(self._display,(0,0,255),False,map_points, int(PIXEL_DENSITY * ROAD_CENTER))
        pygame.draw.rect(self._display,(255,0,0), CAR_RECT)
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        Xct,Yct = self._road[mark][0] - self._location[0] , self._road[mark][1] - self._location[1]
        Xctr,Yctr = Xct * math.cos( math.radians(self._location[2]) ) + Yct * math.sin( math.radians(self._location[2]) ), \
                    Yct * math.cos( math.radians(self._location[2]) ) - Xct * math.sin( math.radians(self._location[2]) )
        X,Y = (Xctr + CAR_X)*PIXEL_DENSITY , (Yctr + CAR_Y)*PIXEL_DENSITY
        #pygame.display.update()
        #pygame.display.flip()

        refrenceYaw = self._road[mark][2] -  self._location[2]
        if refrenceYaw > 180:
            refrenceYaw = refrenceYaw - 360
        if refrenceYaw < -180:
            refrenceYaw = refrenceYaw + 360
        if refrenceYaw % 90 == 0:
            refrenceYaw = 0
        if math.fabs(refrenceYaw) > 60:
            terminal = True
        # compute x error
        x1,y1 = self._road[(mark-1) % len(self._road)][0], self._road[(mark-1)%len(self._road)][1]
        x2,y2 = self._road[(mark + 1) % len(self._road)][0], self._road[(mark + 1) % len(self._road)][1]
        x3,y3 = self._location[0], self._location[1]
        l1 = math.sqrt((x1-x2) **2 + (y1-y2) ** 2)
        l2 = math.sqrt((x3-x2) **2 + (y3-y2) ** 2)
        l3 = math.sqrt((x3-x1) **2 + (y3-y1) ** 2)
        p = (l1 + l2 + l3)/2
        S = math.sqrt(p * (p-l1) * (p-l2) * (p-l3))
        refrenceX = 0
        if l1 != 0:
            refrenceX = S * 2 / l1
        #compute longest distance
        longest = math.fabs(refrenceX) + \
            (CAR_LENGTH/CAR_DUI* math.fabs(math.sin(math.radians(refrenceYaw))) +\
            CAR_WIDTH/CAR_DUI * math.fabs(math.cos(math.radians(refrenceYaw)))) *\
            CAR_DUI *3.5 / 5.5
        if longest >= ROAD_WIDTH/2:
            terminal = True
        reward = math.exp( (-2) * (longest -1.3) )

        reward = math.exp( (-10) * (refrenceX) ) +  (math.exp( 1 -  math.fabs(refrenceYaw) ) - math.exp(0)) + math.exp(-20*math.fabs(self._roll))
        reward = reward / 5
        self._error_dis, self._error_yaw = longest,refrenceYaw
        self._error_dis, self._error_yaw = refrenceX, refrenceYaw
        del map_points
        del map_points_left
        del map_points_right
        del outer_points
        pygame.display.flip()
        #if terminal or self._offroad > 0:
        if terminal:
            reward = -5
        self._terminal = terminal
        return image_data, reward, terminal

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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-m', '--map-name',
        metavar='M',
        default='Town01',
        help='plot the map of the current city (needs to match active map in '
             'server, options: Town01 or Town02)')
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    while True:
        try:

            with make_carla_client(args.host, args.port) as client:
                game = CarlaGame(client, args)
                game.execute()
                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
