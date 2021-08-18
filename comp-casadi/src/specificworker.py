#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
from MPCModel import *
from time import time
import matplotlib.pyplot as plt 
import numpy as np
import warnings  

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel

warnings.simplefilter('ignore', np.RankWarning)

class SpecificWorker(GenericWorker):
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)

        self.controller = MPC()
        self.Period = 500                     # time in ms
        if startup_check:
            self.startup_check()
        else:
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

    def __del__(self):
        console.print('SpecificWorker destructor')

    def setParams(self, params):
        # try:
        #   self.innermodel = InnerModel(params["InnerModelPath"])
        # except:
        #   traceback.print_exc()
        #   print("Error reading config params")
        return True

    @QtCore.Slot()
    def compute(self):
        tic = time()
        print('SpecificWorker.compute...')

        # get current pose in world frame
        currentPose = self.omnirobot_proxy.getBaseState()
        # print(type(currentPose), "\n")
        # print(currentPose)

        rotMat = np.array([
            [cos(-currentPose.alpha), -sin(-currentPose.alpha), 0],
            [sin(-currentPose.alpha),  cos(-currentPose.alpha), 0],
            [0, 0, 1]
        ])

        initialState = ca.DM([currentPose.x, currentPose.z, currentPose.alpha])
        controlState = rotMat @ ca.DM([[currentPose.advVx,
                                        currentPose.advVz, currentPose.rotV]]).T

        # for point stabilization
        # targetState = ca.DM([1800, -200, np.pi/2])

        # calculate mpc in world frame
        controlMPC = self.controller.compute(
            initialState, X_COEFFS, Y_COEFFS, isDifferential=False)
        # apply speed
        vx, vy, w = list(np.array(controlMPC.full()).flatten()
                         )  # TODO: move into class
        self.omnirobot_proxy.setSpeedBase(vx, vy, w)

        # print(f"Time Elapsed = {time() - tic}")
        # print(f"Initial state: {initialState}")
        #print(f"controlState: {controlState}")
        # print(f"Vx: {vx:.2f}, Vy: {vy:.2f}, W: {w:.2f}")

        return True

    def startup_check(self):
        QTimer.singleShot(200, QApplication.instance().quit)

    ######################
    # From the RoboCompLaser you can call this methods:
    # self.laser_proxy.getLaserAndBStateData(...)
    # self.laser_proxy.getLaserConfData(...)
    # self.laser_proxy.getLaserData(...)

    ######################
    # From the RoboCompLaser you can use this types:
    # RoboCompLaser.LaserConfData
    # RoboCompLaser.TData

    ######################
    # From the RoboCompOmniRobot you can call this methods:
    # self.omnirobot_proxy.correctOdometer(...)
    # self.omnirobot_proxy.getBasePose(...)
    # self.omnirobot_proxy.getBaseState(...)
    # self.omnirobot_proxy.resetOdometer(...)
    # self.omnirobot_proxy.setOdometer(...)
    # self.omnirobot_proxy.setOdometerPose(...)
    # self.omnirobot_proxy.setSpeedBase(...)
    # self.omnirobot_proxy.stopBase(...)

    ######################
    # From the RoboCompOmniRobot you can use this types:
    # RoboCompOmniRobot.TMechParams
