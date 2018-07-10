
import sys
sys.path.append("./CubicSpline/")
import math
from proPrint import *
import cubic_spline_planner
import matplotlib.pyplot as plt
import time
import numpy as np

#dependent function
def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile

#dependent function
def calc_speed_profile2(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            speed_profile[i] = target_speed * abs(math.cos(dangle))

    speed_profile[-1] = 0.0
    speed_profile[-2] = 0.0

    return speed_profile

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
        self.spline = False
        self.cx = []
        self.cy = []
        self.cyaw = []
        self.ck = [] # curvature
        self.sp = [] # ref sp
        self.nearestIndex = 0
        self.nearestDis = 0
        self.xref = []

        self.leftcx = []
        self.leftcy = []
        self.rightcy = []
        self.rightcx = []

        self.width = 8.6

        if filename is not None:
            self.readFromFile(filename)
            self.cyaw = smooth_yaw(self.cyaw)

    def setArgs(self, dl = 0.1, spline = False):
        self.dl = dl
        self.spline = spline
        pass

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
                    #print(line)
                contents = line.split('\t')
                #print(contents)
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
                    self.sp.append(float(contents[3]))

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

        if self.spline:
            # calc spline road
            self.cx, self.cy, self.cyaw, self.ck, s = cubic_spline_planner.calc_spline_course(self.cx, self.cy, ds=self.dl )

            # save splined road, named as time
            localtime = time.strftime("%Y-%m-%d_%H:%M:%S.road_py", time.localtime())
            with open(localtime,'w') as f:
                printGreen('Writing splined road to file' + str(localtime) + "......")    
                for cx,cy,cyaw,ck in self.cx, self.cy, self.cyaw, self.ck:
                    f.write(str(cx) + '\t' + str(cy) + '\t' + str(cyaw) + '\t' + str(ck) + '\n')
                printGreen('Done writing splined road to file' + str(localtime) + "!")    


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
    
        dxl = cx - self.cx[ind]
        dyl = cy - self.cy[ind]

        x = math.cos(self.cyaw[ind]) * dxl + math.sin(self.cyaw[ind]) * dyl
        y = math.cos(self.cyaw[ind]) * dyl - math.sin(self.cyaw[ind]) * dxl
    
        self.nearestIndex = ind
        self.nearestDis = y
        #print(cx,cy)
        #print(ind)
        #print(dxl,dyl)
        return ind, y

    
    def calc_ref_trajectory(self, state, HorizonLength):
        #xref = np.zeros((self.NX, self.HorizonLength + 1)) 
        xref = np.zeros(( 4, HorizonLength + 1)) 
        ncourse = len(self.cx)
        cx,cy = state.x,state.y 
        ind, dis = self.findNearest(cx,cy)
    
        xref[0, 0] = self.cx[ind]
        xref[1, 0] = self.cy[ind]
        xref[2, 0] = self.sp[ind]
        xref[3, 0] = self.cyaw[ind]
    
        travel = 0.0 
        DT = 0.2
        DT = 0.1
        #for i in range(T + 1):
        for i in range(HorizonLength + 1): 

            travel += abs(state.v) * DT
            dind = int(round(travel / self.dl))
    
            if (ind + dind) < ncourse:
                xref[0, i] = self.cx[ind + dind]
                xref[1, i] = self.cy[ind + dind]
                xref[2, i] = self.sp[ind + dind]
                xref[3, i] = self.cyaw[ind + dind]
            else:
                xref[0, i] = self.cx[ncourse - 1]
                xref[1, i] = self.cy[ncourse - 1]
                xref[2, i] = self.sp[ncourse - 1]
                xref[3, i] = self.cyaw[ncourse - 1]
        self.xref = xref 
        return xref, ind, dis


    def test(self):
        ax = [35.0, 35.0, 35.0, 33.0, 30.0, 27.0, 23.0, 20.0, 15.0, 10, 11, 14, 16, 12.0, 10.0, 8.0, 6.0, 3.5, 0.0]
        ay = [00.0, 10.0, 25.0, 35.0, 34.0, 32.0, 30.0, 28.0, 29.0, 25.0, 24.0, 20.0, 15.0, 13.0, 10.0, 6.0, 4.0, 0.0 ]
        ax = [0.0, 20.0, 40.0, 60.0, 80.0, 85.0, 90.0, 100.0, 110.0, 105.0, 107.0, 116.0, 108.0, 100.0, 96.0, 90.0, 80.0,60.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 12.0, 15.0, 16.0]
        ay = [0.0, 1.0, -3.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 60.0, 80.0, 95.0, 100.0, 106.0, 110.0, 106.0, 98.0, 90.0, 92.0, 88.0, 80.0, 78.0, 75.0, 70.0, 55.0, 40.0, 20.0]

        self.cx, self.cy, self.cyaw, self.ck, s = cubic_spline_planner.calc_spline_course(
            ax, ay, ds=self.dl)
        self.cyaw = [i - math.pi for i in self.cyaw]
        self.cyaw = smooth_yaw(self.cyaw)
        self.sp = calc_speed_profile2(self.cx, self.cy, self.cyaw, 30)
