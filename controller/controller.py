
import numpy as np
import cvxpy
import matplotlib.pyplot as plt

from PID import PID

import math
import time
import pygame
from pygame.locals import *
import sys
from proPrint import *

#dependent function
def get_nparray_from_matrix(x):
    return np.array(x).flatten()

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
DARKPINK = (255,20,147)
DARKRED = (138,0,0)
PURPLE = (160,32,240)
YELLOW = (255,255,0)
GREEN = (00,255,0)
BLUE = (0,0,255)
LIGHTBLUE = (176,226,255)
ORANGE = (139,69,0)

WIDTH = 1200
HEIGHT = 1000
PIXEL_DENSITY = 2.8

FULL_OFFSET_X = 25
FULL_OFFSET_Y = 25

CAR_WIDTH = 2.4
CAR_LENGTH = 5.5

class Controller:
    def __init__(self,filename = None):
        self.DT = 0.1 # time tick (s)
        #pure pursuit arguments
        self.Kv = 0.3   # look forward gain
        self.Lfc = 6.0    # look ahead distance
        self.PID_throttle = PID(P=0.06, I=0.05, D=0.05)
        self.PID_brake = PID(P=0.07, I=0.05, D=0.02)
        self.WB = 3.6     # wheel base

        self.vehicle = Vehicle()
        self.road = Road()
        if filename is not None:
            self.road = Road(filename)
        self.state = State()

        #init pygame
        pygame.init()
        self._display = pygame.display.set_mode( (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._line = pygame.display.set_mode( (WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.flip()
        self.map_points = self._generate_pointlist()
        self.trajectory = []

        self.logfile = open(time.strftime("./log/%Y-%m-%d_%H:%M:%S", time.localtime()), 'w')

    def pid_speed(self):
        self.PID_throttle.SetPoint = self.state.setV
        self.PID_brake.SetPoint = self.state.setV
        throttle = max(0,min(1, self.PID_throttle.update(self.state.v)))
        brake = min(0,max(-1, self.PID_brake.update(self.state.v)))
        if self.state.setV > self.state.v:
            return throttle, 0
        return 0,brake * -1
        
    def get_aheaddis(self):
        return max(math.log1p(self.state.v) + 3/4 * self.state.v + 2 , self.Lfc )
        #return self.Kv * self.state.v + self.Lfc

    def pp_steer(self):
        ind,dis = self.road.findNearest(self.state.x,self.state.y,start=self.road.nearestIndex, num=100)
        self.state.setV = self.road.sp[ind]
        self.state.errorDis = dis
        self.state.errorYaw = self.state.yaw - self.road.cyaw[ind]
        #print(dis)
        print(self.state.errorYaw)
        if abs(dis) > 20:
            ind,dis = self.road.findNearest(self.state.x,self.state.y)

        mark = ind
        #Lf = self.get_aheaddis()
        Lf = max( math.log1p(self.state.v) + 3/4 * self.state.v + 2, self.Lfc)
        L = 0
        while Lf > L:
            L += math.sqrt(  (self.road.cx[(ind + 1)%len(self.road.cx)] - self.road.cx[ind%len(self.road.cx)])**2 +\
                             (self.road.cy[(ind + 1)%len(self.road.cy)] - self.road.cy[ind%len(self.road.cy)])**2  )
            ind +=1
        target = ind % len(self.road.cx)
        tx,ty,tyaw,tsp = self.road.cx[target], self.road.cy[target], self.road.cyaw[target], self.road.sp[target]

        alpha1 = pi_2_pi( math.atan2(ty - self.state.y, tx - self.state.x) )
        alpha = pi_2_pi(( alpha1 - pi_2_pi(self.state.yaw) ))

        delta = math.atan2(2.0 * self.WB * math.sin(alpha) / Lf, 1.0)
        delta /= (math.pi * 1 )
        if delta > 0:
            delta = (1 + delta) ** 2 - 1
        else:
            delta = 1 - (delta - 1)**2
        delta = max(-1, min(1, delta))
        
        return delta

    def pp_control(self):
        steer = self.pp_steer()
        throttle, brake = self.pid_speed()
        return throttle, brake, steer

    def stanely_steer(self):
        # calc frant axle position
        fx = self.state.x + self.WB * math.cos(self.state.yaw)
        fy = self.state.y + self.WB * math.sin(self.state.yaw)
        ind,dis = self.road.findNearest(fx,fy,start=self.road.nearestIndex, num=100)
        self.state.setV = self.road.sp[ind]
        self.state.errorDis = dis
        self.state.errorYaw = self.state.yaw - self.road.cyaw[ind]
        if abs(dis) > 20:
            ind,dis = self.road.findNearest(self.state.x,self.state.y)

        theta_e = pi_2_pi(self.road.cyaw[ind] - self.state.yaw) * 0.7
        theta_d = math.atan2(-0.5 * dis, self.state.v)
        delta = theta_e + theta_d

        return delta

    def stanely_control(self):
        steer = self.stanely_steer()
        throttle, brake = self.pid_speed()
        return throttle, brake, delta

    def pid_steer(self):
        pass

    def log(self):
        message = str(self.state.x) + '\t' + str(self.state.y) + '\t' + str(self.state.yaw) + '\t' + str(self.state.roll) + '\t' +\
                  str(self.state.errorDis) + '\t' + str(self.state.errorYaw) + '\n'
        self.logfile.write(message)
        pass

    def render(self):
        self._line.fill(BLACK)
        self._display.fill(BLACK)
        self.trajectory.append(( int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y)  ) )
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[2:]
        #if True:
        pygame.draw.lines(self._line,GREEN,False,self.map_points, 2)
        pygame.draw.circle(self._line, (255,255,255), [int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y)], 3)
        pygame.draw.lines(self._line, RED, False, \
                                    [ ( int(self.state.x * PIXEL_DENSITY + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + FULL_OFFSET_Y) ),\
                                     ( int(self.state.x * PIXEL_DENSITY + math.cos(self.state.yaw)* 20 + FULL_OFFSET_X), int(self.state.y * PIXEL_DENSITY + math.sin(self.state.yaw)*20 + FULL_OFFSET_Y) )] ,3)
        if len(self.trajectory) > 3:
            pygame.draw.lines(self._line, ORANGE,False,self.trajectory, 2)
        pass

        for event in pygame.event.get():
            pass
        keys = pygame.key.get_pressed()
        pygame.display.flip()
        pygame.display.update()

    def _generate_pointlist(self):
        map_points = []
        for (cx,cy) in zip(self.road.cx,self.road.cy):
            map_points.append( (int(cx * PIXEL_DENSITY + FULL_OFFSET_X),int(cy * PIXEL_DENSITY + FULL_OFFSET_Y)) )
        return map_points

    def update_state(self,state, a, delta):
        if delta >= self.vehicle.MAX_STEER:
            delta = self.vehicle.MAX_STEER
        elif delta <= -self.vehicle.MAX_STEER:
            delta = -self.vehicle.MAX_STEER
    
        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.vehicle.WB * math.tan(delta) * self.DT
        state.v = state.v + a * self.DT
    
        if state. v > self.vehicle.MAX_SPEED:
            state.v = self.vehicle.MAX_SPEED
        elif state. v < self.vehicle.MIN_SPEED:
            state.v = self.vehicle.MIN_SPEED
    
        return state

class Vehicle:
    def __init__(self):
        self.LENGTH = 4.5  # [m]
        self.WIDTH = 2.0  # [m]
        self.BACKTOWHEEL = 1.0  # [m]
        self.WHEEL_LEN = 0.3  # [m]
        self.WHEEL_WIDTH = 0.2  # [m]
        self.TREAD = 0.7  # [m]

        self.MAX_STEER = math.radians(45.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = math.radians(30.0)  # maximum steering speed [rad/s]
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

#dependent function
def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]
        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]
    return yaw

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle

class Road:

    def __init__(self,filename = None):
        self.dl = 0.1  # course tick
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.ck = [] # curvature
        self.sp = [] # ref sp
        self.nearestIndex = None
        self.nearestDis = None
        self.xref = []

        self.leftcx = []
        self.leftcy = []
        self.rightcy = []
        self.rightcx = []

        self.width = 8.6

        if filename is not None:
            self.readFromFile(filename)
            self.cyaw = smooth_yaw(self.cyaw)

    def readFromFile(self,filename):
        self.cx = []
        self.cy = []
        # read from map file and assign road point value
        with open(filename,'r') as f:
            printGreen('Reading map file........')    
            lineNum = 0
            line = f.readline()
            lineNum += 1
            while line:
                if lineNum % 5000 == 0:
                    printGreen('Read ' + str(lineNum) + ' lines......' )    
                contents = line.split('\t')
                if len(contents) < 4:
                    line = f.readline()
                    lineNum += 1
                    continue
                #assert len(contents) > 1,'map file is not correct at line ' + str(lineNum) + '!'
                
                try:
                    x,y,yaw = float(contents[0]), float(contents[1]), pi_2_pi(math.radians(float(contents[2])))
                    self.cx.append(x)
                    self.cy.append(y)
                    #yaw = pi_2_pi(math.radians(float(contents[2])))
                    #self.cyaw.append(float(contents[2]))
                    self.cyaw.append(yaw)
                    self.sp.append(float(contents[3])/3.6)

                    self.leftcx.append(( x + self.width/2 * math.sin(yaw)) )
                    self.leftcy.append(( y - self.width/2 * math.cos(yaw)) )

                    self.rightcx.append(( x - self.width/2 * math.sin(yaw)) )
                    self.rightcy.append(( y + self.width/2 * math.cos(yaw)) )

                    line = f.readline()
                    lineNum += 1
                except SyntaxError:
                    printRed("The format is incorrect at line " + str(lineNum))
                else:
                    continue
            printGreen('Done reading map file!')    


    def findNearest(self,cx,cy,start=None, num=None):
        dx = []
        dy = []
        if num is not None and start is not None:
            dx = [cx - icx for icx in self.cx[start:start+num] ]
            dy = [cy - icy for icy in self.cy[start:start+num] ]
            if start + num > len(self.cx):
                dx = dx + [cx - icx for icx in self.cx[0: start + num - len(self.cx)] ]
                dy = dy + [cy - icy for icy in self.cy[0: start + num - len(self.cy)] ]
        else:
            dx = [cx - icx for icx in self.cx[ : ] ]
            dy = [cy - icy for icy in self.cy[ : ] ]
        pass

        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        ind = d.index(min_d)
        if start is not None:
            ind += start
            ind %= len(self.cx)
    
        dxl = cx - self.cx[ind]
        dyl = cy - self.cy[ind]

        x = math.cos(self.cyaw[ind]) * dxl + math.sin(self.cyaw[ind]) * dyl
        y = math.cos(self.cyaw[ind]) * dyl - math.sin(self.cyaw[ind]) * dxl
    
        self.nearestIndex = ind
        self.nearestDis = y
        return ind, y

    
    def calc_ref_trajectory(self, ind, ahead,behind):
        ind %= len(self.cx)
        cx = []
        cy = []
        cyaw = []
        sp = []
        if ind - behind >= 0:
            cx = self.cx[ind-behind:ind]
            cy = self.cy[ind-behind:ind]
            cyaw = self.cyaw[ind-behind:ind]
            sp = self.sp[ind-behind:ind]
        else:
            cx =     self.cx[(ind-behind)%len(self.cx):len(self.cx)] + self.cx[0:ind]
            cy =     self.cy[(ind-behind)%len(self.cx):len(self.cx)] + self.cy[0:ind]
            cyaw = self.cyaw[(ind-behind)%len(self.cx):len(self.cx)] + self.cyaw[0:ind]
            sp =     self.sp[(ind-behind)%len(self.cx):len(self.cx)] + self.sp[0:ind]
        if ind + ahead < len(self.cx):
            cx = cx + self.cx[ind:ind+ahead]
            cy = cy + self.cy[ind:ind+ahead]
            sp = sp + self.sp[ind:ind+ahead]
            cyaw = cyaw + self.cyaw[ind:ind+ahead]
        else:
            cx = cx + self.cx[ind: ind + ahead] + self.cx[0: (ind+ahead) % len(self.cx)]
            cy = cy + self.cy[ind: ind + ahead] + self.cy[0: (ind+ahead) % len(self.cy)]
            cyaw = cyaw + self.cyaw[ind: ind + ahead] + self.cyaw[0: (ind+ahead) % len(self.cyaw)]
            sp = sp + self.sp[ind: ind + ahead] + self.sp[0: (ind+ahead) % len(self.sp)]
        
        return cx,cy,cyaw,sp
