import os
import numpy as np
import time
import pybullet_data
import random
import glob

from robot import KUKAROBOTIQ, KUKASAKE

class ENVIRONMENT:
    def __init__(self,
                 bc,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=80,
                 max_steps=64,  # 8
                 dv=0.01,  # 0.06
                 block_random=0.3,
                 camera_random=0,
                 width=48,
                 height=48,
                 num_objects=1):
        """Initializes the KukaDiverseObjectEnv.
    Args:
      urdf_root: The diretory from which to load environment URDF's.
      action_repeat: The number of simulation steps to apply for each action.
      max_steps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      block_random: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      camera_random: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      num_objects: The number of objects in the bin.
    """
        self.bc = bc  # simulation server
        self.time_step = 1. / 240.
        self.urdf_root = urdf_root
        self.action_repeat = action_repeat
        self.observation = []
        self.max_steps = max_steps
        self.terminated = 0
        self.dv = dv
        self.block_random = block_random
        self.camera_random = camera_random
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.num_actions = 7

        # rendering
        self.cid = self.bc.connect(self.bc.SHARED_MEMORY)
        if self.cid < 0:
            self.cid = self.bc.connect(self.bc.GUI)
        self.bc.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])

        self.action_space = np.array([[0, -self.dv, self.dv, 0, 0, 0, 0],
                                      [0, 0, 0, -self.dv, self.dv, 0, 0],
                                      [-self.dv, -self.dv, -self.dv, -self.dv, -self.dv, -self.dv, -self.dv],
                                      [0, 0, 0, 0, 0, -0.25, 0.25],
                                      [0, 0, 0, 0, 0, 0, 0]])

    def camera_reset(self):
        look = [0.23, 0.2, 0.73]
        distance = 1.
        pitch = -56 + self.camera_random * np.random.uniform(-3, 3)
        yaw = 245 + self.camera_random * np.random.uniform(-3, 3)
        roll = 0
        self.view_matrix = self.bc.computeViewMatrixFromYawPitchRoll(look, distance, yaw, pitch, roll, 2)
        fov = 20. + self.camera_random * np.random.uniform(-2, 2)
        aspect = self.width / self.height
        near, far = 0.01, 10
        self.proj_matrix = self.bc.computeProjectionMatrixFOV(fov, aspect, near, far)

    def reset(self):
        self.camera_reset()
        self.attempted_grasp = False
        self.env_step = 0
        self.terminated = 0

        self.bc.resetSimulation()
        self.bc.setPhysicsEngineParameter(numSolverIterations=150)
        self.bc.setTimeStep(self.time_step)
        self.bc.setGravity(0, 0, -10)
        self.bc.loadURDF(os.path.join(self.urdf_root, "plane.urdf"), [0, 0, -0.63])
        self.bc.loadURDF(os.path.join(self.urdf_root, "table/table.urdf"),
                         [0.50, 0.00, -0.63], [0, 0, 0, 1])
        self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "tray/tray.urdf"),
                         [0.64, 0.00, 0.00], [0, 0, 1, 0])
        self.robot = KUKASAKE(self.bc, [0, 0, 0], [0, 0, 0, 1])
        self.bc.stepSimulation()

        # Choose the objects in the bin.
        self.object_ids = self.randomly_place_objects(self.get_random_object(self.num_objects))
        self.observation = self.get_observation()
        return np.array(self.observation)

    def randomly_place_objects(self, urdfList):
        objectUids = []
        for urdf_name in urdfList:
            xpos = 0.4 + self.block_random * random.random()
            ypos = self.block_random * (random.random() - .5)
            angle = np.pi / 2 + self.block_random * np.pi * random.random()
            orn = self.bc.getQuaternionFromEuler([0, 0, angle])
            urdf_path = os.path.join(self.urdf_root, urdf_name)
            uid = self.bc.loadURDF(urdf_path, [xpos, ypos, .15], [orn[0], orn[1], orn[2], orn[3]])
            objectUids.append(uid)
            for _ in range(500):
                self.bc.stepSimulation()
        return objectUids

    def get_observation(self):
        img_arr = self.bc.getCameraImage(width=self.width,
                                         height=self.height,
                                         viewMatrix=self.view_matrix,
                                         projectionMatrix=self.proj_matrix)
        np_img_arr = np.reshape(img_arr[2], (self.height, self.width, 4))
        return np_img_arr[:, :, :3]

    def step(self, action):
        action = self.action_space[:, action]

        self.env_step += 1
        self.robot.applyAction(action)
        for _ in range(self.action_repeat):
            self.bc.stepSimulation()
            time.sleep(self.time_step)
            if self.termination():
                break

        # If we are close to the bin, attempt grasp.
        state = self.bc.getLinkState(self.robot.robot_id, self.robot.ee_id)
        end_effector_pos = state[0]
        extern = state[1]

        if end_effector_pos[2] <= 0.05 + self.robot.gripper_z_offset:
            finger_angle = 0.3
            for i in range(1000):
                if i < 500:
                    grasp_action = [0, 0, 0, 0, finger_angle]
                else:
                    grasp_action = [0, 0, 0.001, 0, finger_angle]
                self.robot.applyAction(grasp_action)
                self.bc.stepSimulation()
                time.sleep(self.time_step)
                finger_angle -= 0.3 / 100.
                if finger_angle < 0:
                    finger_angle = 0
            self.attempted_grasp = True

        observation = self.get_observation()
        done = self.termination()
        reward = self.reward()
        debug = {'grasp_success': self.grasp_success}

        return observation, reward, done, debug, \
            end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], \
            extern[0], extern[1], extern[2], extern[3]

    def reward(self):
        self.grasp_success = 0
        for uid in self.object_ids:
            obj_pos, _ = self.bc.getBasePositionAndOrientation(uid)
            if obj_pos[2] > 0.2:
                self.grasp_success = 1
                return self.grasp_success
        return self.grasp_success

    def termination(self):
        return self.attempted_grasp or self.env_step >= self.max_steps

    def get_random_object(self, num_objects):
        urdf_pattern = os.path.join(self.urdf_root, 'random_urdfs/*[1-9]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
        selected_objects_filenames = []
        for object_index in selected_objects:
            selected_objects_filenames += [found_object_directories[object_index]]
        return selected_objects_filenames
