#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   taskFirst.py
@Time    :   2021/06/10 5:00:00
@Author  :   JC Zhang
@Version :   1.6
@Contact :   jczhang@live.it
@License :   GUN
@Desc    :   None
'''

# here put the import lib
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from datetime import datetime
import os
import linecache
import pybullet as p
import pybullet_data
import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import threading
from numpy import arange
import logging
import math
from termcolor import colored
import threading
from rich.console import Console

console = Console()


# 一些变量 ######k
LOGGING_LEVEL = logging.INFO
# is_render=False
# is_good_view=False   #这个的作用是在step时加上time.sleep()，把机械比的动作放慢，看的更清，但是会降低训练速度
#########################

# logging.basicConfig(
#     level=LOGGING_LEVEL,
#     format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     filename='../logs/reach_env.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
#     filemode='w')
# logger = logging.getLogger(__name__)
# env_logger=logging.getLogger('env.py')

# logging模块的使用
# 级别                何时使用
# DEBUG       细节信息，仅当诊断问题时适用。
# INFO        确认程序按预期运行
# WARNING     表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
# ERROR       由于严重的问题，程序的某些功能已经不能正常执行
# CRITICAL    严重的错误，表明程序已不能继续执行


class KukaReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 1000

    def __init__(self, is_render=False, is_good_view=False, t=0, ot=0, filenameLog=""):
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.t = t
        self.ot = ot
        self.filenameLog = filenameLog

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs = 0.0
        self.x_high_obs = 0.5
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_mid = (self.x_low_obs + self.x_high_obs) / 2
        self.y_mid = 0

        self.x_low_action = 0.0
        self.x_high_action = 1
        self.y_low_action = -0.6
        self.y_high_action = 0.6
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        p.resetDebugVisualizerCamera(cameraDistance=2,
                                     cameraYaw=0,
                                     cameraPitch=-80,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
            high=np.array([
                self.x_high_action,
                self.y_high_action,
                self.z_high_action]),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs]),
            dtype=np.float32)
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.seed()

        self.reset()

    def addObject2(self):
        self.ranX2 = random.uniform(0, 1)
        self.ranY2 = random.uniform(-0.6, 0.6)
        self.object_id2 = p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
                                     basePosition=[self.ranX2,
                                                   self.ranY2,
                                                   0.01])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def addObstacle(self):
        self.obstacle_id = p.loadURDF("./data/cube.urdf",
                                      basePosition=[random.uniform(0, self.ranXP),
                                                    random.uniform(-0.6, 0.6),
                                                    0.01],
                                      globalScaling=0.1,
                                      flags=p.URDF_USE_SELF_COLLISION)

    def addObstacle2(self):
        self.obstacle_id = p.loadURDF("./data/cube.urdf",
                                      basePosition=[random.uniform(0, self.ranX2),
                                                    random.uniform(-0.6, 0.6),
                                                    0.01],
                                      globalScaling=0.1,
                                      flags=p.URDF_USE_SELF_COLLISION)

    def addObstacleOut(self):
        self.obstacle_id = p.loadURDF("./data/cube.urdf",
                                      basePosition=[random.uniform(0, self.ranXP),
                                                    random.uniform(-0.6, 0.6),
                                                    0.01],
                                      globalScaling=0.1,
                                      flags=p.URDF_USE_SELF_COLLISION)
        '''self.obstacle_id = p.loadURDF("./data/cube.urdf",
                                      basePosition=[random.uniform(self.x_low_action, (self.x_low_action + self.x_high_action) / 2),
                                                    random.uniform(self.y_low_action, 0),
                                                    0.01],
                                      globalScaling=0.1,
                                      flags=p.URDF_USE_SELF_COLLISION)'''

    def reset(self):
        # p.connect(p.GUI)
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        p.setGravity(0, 0, -10)

        # 外界大的边界 蓝色
        # 竖线

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_action, self.y_low_action, 0],
            lineToXYZ=[self.x_low_action, self.y_high_action, 0],
            lineColorRGB=[1, 0, 0]
        )

        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_action, self.y_low_action, 0],
            lineToXYZ=[self.x_high_action, self.y_high_action, 0],
            lineColorRGB=[0, 0, 255],
            lineWidth=2
        )
        # 横线

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_action, self.y_low_action, 0],
            lineToXYZ=[self.x_high_action, self.y_low_action, 0],
            lineColorRGB=[0, 0, 255]
        )

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_action, self.y_high_action, 0],
            lineToXYZ=[self.x_high_action, self.y_high_action, 0],
            lineColorRGB=[0, 0, 255]
        )

        #　内部小边界

        # 两条较长的横线
        '''
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - 0.2, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs + 0.2, self.y_low_obs, 0],
            lineColorRGB=[255, 97, 0]
        )

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs - 0.2, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs + 0.2, self.y_high_obs, 0],
            lineColorRGB=[255, 97, 0]
        )
        # 一条较长的竖线
        p.addUserDebugLine(
            lineFromXYZ=[(self.x_low_obs + self.x_high_obs) / 2, self.y_low_action - 0.2, 0],
            lineToXYZ=[(self.x_low_obs + self.x_high_obs) / 2, self.y_high_action + 0.2, 0],
            lineColorRGB=[255, 96, 0]
        )
        '''

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)

        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.3, 0, -0.65 * 2], globalScaling=2)
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        # object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])

        self.ranX = random.uniform(0, 1)
        self.ranY = random.uniform(-0.6, 0.6)

        self.ranXP = self.ranX
        self.ranYP = self.ranY
        '''self.ranXP = random.uniform(self.x_low_obs, (self.x_low_action + self.x_high_action) / 2)
        self.ranYP = random.uniform(self.y_low_action, 0)
        '''
        self.object_id = p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
                                    basePosition=[self.ranX,
                                                  self.ranY,
                                                  0.01])

        '''if (self.ot == 0):
            self.object_id = p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
                                        basePosition=[self.ranX,
                                                      self.ranY,
                                                      0.01])
        elif (self.ot == 1):
            self.object_id = p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"),
                                        basePosition=[self.ranXP,
                                                      self.ranYP,
                                                      0.01])'''

        self.num_joints = p.getNumJoints(self.kuka_id)

        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        #print(p.getLinkState(self.kuka_id, self.num_joints - 1)[4])
        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        # logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        return np.array(self.object_pos).astype(np.float32)
        # return np.array(self.robot_pos_obs).astype(np.float32)
    '''
    def getInversePoisition(self, Uid, position_desired, orientation_desired=[]):
        joints_info = []
        joint_damping = []
        joint_ll = []
        joint_ul = []
        useOrientaion = len(orientation_desired)
        # for i in range(24):
        #    joints_info.append(p.getJointInfo(Uid, i))
        kukaEndEffectorIndex = 6
        #numJoints = p.getNumJoints(Uid)
        useNullSpace = 1
        ikSolver = 1
        pos = [position_desired[0], position_desired[1], position_desired[2]]
        if useOrientaion:
            orn = p.getQuaternionFromEuler([orientation_desired[0], orientation_desired[1], orientation_desired[2]])
        if (useNullSpace == 1):
            if (useOrientaion == 1):
                jointPoses = p.calculateInverseKinematics(Uid, kukaEndEffectorIndex, pos, orn)
            else:
                jointPoses = p.calculateInverseKinematics(Uid, kukaEndEffectorIndex, pos, lowerLimits=joint_ll, upperLimits=joint_ul)
        else:
            if (useOrientaion == 1):
                jointPoses = p.calculateInverseKinematics(Uid, kukaEndEffectorIndex, pos, orn, solver=ikSolver, maxNumIterations=10, residualThreshold=.01)
            else:
                jointPoses = p.calculateInverseKinematics(Uid, kukaEndEffectorIndex, pos, solver=ikSolver)
            p.getJointInfo()
        return jointPoses

    def get_to_place(self, pos):
        orn = []
        jointPoses = self.getInversePoisition(self.kuka_id, pos)
        for i in range(7):
            p.setJointMotorControl2(bodyIndex=self.kuka_id,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i - 3],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        return
        '''

    def write_log(self, pos):
        filename = self.filenameLog
        # print(datetime.now().second)
        x = pos[0]
        y = pos[1]
        z = pos[2]
        out = "(" + str(x) + "," + str(y) + "," + str(z) + ")"
        f = open(filename, "a")
        print(out, file=f)
        f.close()

    def clue_debugLine(self, pos):

        objectState = np.array(p.getBasePositionAndOrientation(self.object_id)[0]).astype(np.float32)
        dx = objectState[0]
        dy = objectState[1]
        dz = objectState[2]

        x = pos[0]
        y = pos[1]
        z = pos[2]

        cnt = 0

        if (dx <= x and dy <= y):
            str = "Left and Up"
            print(str)
            '''
            text1 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=1.2,
            )
            cnt = text1
            '''
        if (dx <= x and dy >= y):
            str = "Left and Down"
            print(str)
            '''
            text2 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=0.2,
            )
            cnt = text2
            '''

        if (dx >= x and dy <= y):
            str = "Right and Up"
            print(str)
            '''
            text3 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=0.2,
            )
            cnt = text3
            '''

        if (dx >= x and dy >= y):
            str = "Right and Down"
            '''
            text4 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=1.2,
            )
            cnt = text4
            '''

        return cnt

    """
    def clue(self):
        objectState = np.array(p.getBasePositionAndOrientation(self.object_id)[0]).astype(np.float32)
        dx = objectState[0]
        dy = objectState[1]
        dz = objectState[2]

        kukaState = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        x = kukaState[0]
        y = kukaState[1]
        z = kukaState[2]

        cnt = 0

        if (dx <= x and dy <= y):
            str = "Left and Up"
            print(str)
            '''
            text1 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=1.2,
            )
            cnt = text1
            '''
        if (dx <= x and dy >= y):
            str = "Left and Down"
            print(str)
            '''
            text2 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=0.2,
            )
            cnt = text2
            '''

        if (dx >= x and dy <= y):
            str = "Right and Up"
            print(str)
            '''
            text3 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=0.2,
            )
            cnt = text3
            '''

        if (dx >= x and dy >= y):
            str = "Right and Down"
            '''
            text4 = p.addUserDebugText(
                text=str,
                textPosition=[x, y, z],
                textColorRGB=[0, 1, 0],
                textSize=1.2,
            )
            cnt = text4
            '''

        return cnt
        """

    def debugLine(self, pos1, pos2):
        p.addUserDebugLine(pos1, pos2, [0, 1, 0], lineWidth=2)
        self.clue_debugLine(pos1)
        self.write_log(pos1)
        return self._reward(pos1)

    def step(self, action):

        self.new_robot_pos = action
        # print(action)

        # for i in range(1, 4000):
        # p.stepSimulation()
        # self.get_to_place(action)

        self.new_robot_pos = action
        print(action)
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,)
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.kuka_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.robot_joint_positions[i],
                targetVelocity=0,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )


        havePrevPos = 0

        for i in range(200):
            p.stepSimulation()
            self.setCameraPicAndGetPic(self.kuka_id)

            # draw!

            Pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
            if havePrevPos == 1:
                p.addUserDebugLine(prevPos, Pos, [0, 1, 0], lineWidth=2)
            prevPos = Pos

            havePrevPos = 1




            time.sleep(1 / 2000)
            # for j in range(2):
            #     self.write_log()
        # 40 hz

        if self.is_good_view:
            time.sleep(0.05)

        return self._reward(action)

    def setCameraPicAndGetPic(self, robot_id : int, width : int = 224, height : int = 224, physicsClientId : int = 0):
        """
        给合成摄像头设置图像并返回robot_id对应的图像
        摄像头的位置为miniBox前头的位置
        """
        basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
        # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
        matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])              # 变换后的x轴
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])              # 变换后的z轴


        basePos = np.array(basePos)
        # 摄像头的位置
        BASE_RADIUS = 1
        BASE_THICKNESS = 1
        cameraPos = basePos + BASE_RADIUS * tx_vec + 0.5 * BASE_THICKNESS * tz_vec
        targetPos = cameraPos + 1 * tx_vec
        kuka_state = self.getRobotState()
        x = kuka_state[0]
        y = kuka_state[1]
        z = kuka_state[2]


        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
                [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  # the direction is from the light source position to the origin of the world frame.
        }

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[x, y, z],
            distance=.5,
            yaw=0,
            pitch=-70,
            roll=0,
            upAxisIndex=2)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
                   self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=width, height=height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            physicsClientId=physicsClientId
        )

        return width, height, rgbImg, depthImg, segImg


    def getRobotState(self):
        return p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
    

    def _reward(self, action):

        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        # self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)
        #
        self.robot_state = action
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(np.float32)

        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy)
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        '''
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        '''

        terminated = 0

        if self.distance < 0.1:
            reward = 1
            self.terminal = 1
        else:
            reward = 0
            self.terminal = 0

        '''
        if terminated:
            reward = -0.1
            self.terminated = True
        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.distance < 0.1:
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False
        '''

        info = {'distance:', self.distance}
        # self.observation=self.robot_state
        self.observation = self.object_state
        return np.array(self.observation).astype(
            np.float32), reward, self.terminated, info, np.array(self.robot_state).astype(
            np.float32)

    def close(self):
        p.disconnect()


def getContent(cnt, filename='pos.txt'):
    ini = linecache.getline(filename, cnt)
    ini = ini[1:]
    ini = ini[:-2]
    print(ini)
    a, b, c = ini.split(",")
    c = 0.0
    return float(a), float(b), float(c)


def getCount(filename='pos.txt'):
    cnt = len(open(filename, 'rU').readlines())
    return cnt


def inter():
    print("Time so Long!")
    os._exit(0)


def currentWork(k, folder):
    global timer
    timer = threading.Timer(600.0, inter)

    timer.start()
    listLogFile = ['log1.txt', 'log2.txt', 'log3.txt', 'log4.txt']
    files = folder + "/" + listLogFile[k - 1]
    global origin
    origin = files
    work(at=k, fileName=folder + "/" + listLogFile[k - 1])
    evalute(fileName=origin)


def stepEvalute():
    k = 20
    k_current = evalute(origin)
    score = max((k - k_current), 0) / k
    return score


def evalute(fileName):
    human_x = []
    human_y = []
    for i in range(1, getCount(filename=fileName)):
        x, y, z = getContent(i, filename=fileName)
        human_x.append(x)
        human_y.append(y)
    robot_x = []
    robot_y = []
    for i in range(1, getCount(filename='logRobot.txt') + 1):
        x, y, z = getContent(i, filename='logRobot.txt')
        robot_x.append(x)
        robot_y.append(y)

    sx1 = np.array(human_x)
    sy1 = np.array(human_y)
    sx2 = np.array(robot_x)
    sy2 = np.array(robot_y)

    '''path1 = dtw.warping_path(sx1, sx2)
    path2 = dtw.warping_path(sy1, sy2)
    dtwvis.plot_warping(sx1, sx2, path1, filename="./evalute/pic1/warpx.png")
    path1 = dtw.warping_path(sy1, sy2)
    dtwvis.plot_warping(sy1, sy2, path2, filename="./evalute/pic1/warpy.png")'''

    # type two evalute
    random.seed(1)
    maxLen = max(len(sx1), len(sx2))
    for idx in range(maxLen):
        if random.random() < 0.05:
            sx2[idx] += (random.random() - 0.5) / 2
    d, paths = dtw.warping_paths(sx1, sx2, window=25, psi=2)
    best_paths = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(sx1, sx2, paths, best_paths, filename="./evalute/pic2/wrapx.png")
    for idx in range(maxLen):
        if random.random() < 0.05:
            sy2[idx] += (random.random() - 0.5) / 2
    d, paths = dtw.warping_paths(sy1, sy2, window=25, psi=2)
    best_paths = dtw.best_path(paths)
    dtwvis.plot_warpingpaths(sy1, sy2, paths, best_paths, filename="./evalute/pic2/wrapy.png")

    return d


def work(at, fileName):

    rangeTime = 1
    te = 0

    if (at == 1):
        env = KukaReachEnv(is_render=True, is_good_view=False, t=at, filenameLog=fileName)
        for i in range(rangeTime):
            loop = 1

            for j in range(1, getCount()):
                action = np.array(getContent(j))
                action2 = np.array(getContent(j + 1))
                
                '''tk1 = threading.Thread(target=env.debugLine, args=(action, action2))
                tk2 = threading.Thread(target=env.step, args=(action))
                tk1.start()
                tk2.start()
                tk1.join()
                tk2.join()'''
                env.step(action)
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)

                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    te = 1
                    break
        console.print("Finished", style="bold cyan")
        if (te == 1):
            console.print("Successfully reach target", style="bold green")
            te = 0
        else:
            console.print("Failed reach target", style="bold red")

    elif (at == 2):
        env = KukaReachEnv(is_render=True, is_good_view=False, t=at, filenameLog=fileName)
        for i in range(rangeTime):
            loop = 1
            # TODO  add multi coordinates support
            for j in range(1, getCount("pos.txt")):
                action = np.array(getContent(j))
                action2 = np.array(getContent(j + 1))
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)
                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    te = 1
                    break
        console.print("Finished Find first object", style="bold cyan")
        if (te == 1):
            console.print("Successfully reach target 1", style="bold green")
            te = 0
        else:
            console.print("Failed reach target 1", style="bold red")
        env.addObject2()
        for i in range(rangeTime):
            loop = 1
            # TODO  add multi coordinates support
            for j in range(1, getCount("pos2.txt")):
                action = np.array(getContent(j, "pos2.txt"))
                action2 = np.array(getContent(j + 1, "pos2.txt"))
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)
                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    break
        console.print("Finished Find second object", style="bold cyan")
        if te == 1:
            console.print("Successfully reach target 2", style="bold green")
            te = 0
        else:
            console.print("Failed reach target 2", style="bold red")

    elif (at == 3):
        env = KukaReachEnv(is_render=True, is_good_view=False, t=at, filenameLog=fileName)
        env.addObstacle()
        for i in range(rangeTime):
            loop = 1

            for j in range(1, getCount()):
                action = np.array(getContent(j))
                action2 = np.array(getContent(j + 1))
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)
                # obs, reward, done, info, robot_obs = env.step(action)
                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    te = 1
                    break
        console.print("Finished", style="bold cyan")
        if te == 1:
            console.print("Successfully reach target bypass 1 obstacle", style="bold green")
            te = 0
        else:
            console.print("Failed reach target bypass 1 obstacle", style="bold red")

    elif (at == 4):
        env = KukaReachEnv(is_render=True, is_good_view=False, t=at, filenameLog=fileName)
        for i in range(rangeTime):
            loop = 1
            # TODO  add multi coordinates support
            for j in range(1, getCount("pos.txt")):
                action = np.array(getContent(j))
                action2 = np.array(getContent(j + 1))
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)
                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    te = 1
                    break
        console.print("Finished ", style="bold cyan")
        if te == 1:
            console.print("Successfully reach target bypass first obstacle", style="bold green")
            te = 0
        else:
            console.print("Failed reach target bypass first obstacle", style="bold red")

        env.addObject2()
        env.addObstacle2()

        for i in range(rangeTime):
            loop = 1
            # TODO  add multi coordinates support
            for j in range(1, getCount("pos2.txt")):
                action = np.array(getContent(j, "pos2.txt"))
                action2 = np.array(getContent(j + 1, "pos2.txt"))
                obs, reward, done, info, robot_obs = env.debugLine(action, action2)
                # print(colored("info={}".format(info),"cyan"))
                if done == 1:
                    print("done")
                    break
        console.print("Finished ", style="bold cyan")
        if te == 1:
            console.print("Successfully reach target bypass second obstacle", style="bold green")
            te = 0
        else:
            console.print("Failed reach target bypass second obstacle", style="bold red")


if __name__ == '__main__':
    # load file
    inS = input()
    inS = int(inS)
    strFolder = "log" + "/" + datetime.now().strftime("%Y%m%d_%H:%M:%S") + "Task" + str(inS)

    os.makedirs(strFolder)
    currentWork(inS, strFolder)
