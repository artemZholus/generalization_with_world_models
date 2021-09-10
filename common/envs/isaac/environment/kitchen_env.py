import gym
import numpy as np
import cv2
import os
import yaml

import carb
import omni.kit.app
import omni.kit.editor as ed

from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
from omni.physx.scripts import physicsUtils, utils

from omni.isaac.dynamic_control import _dynamic_control

from pxr import UsdGeom, Gf, Sdf, Usd, PhysxSchema, PhysicsSchema, PhysicsSchemaTools, Semantics
from ..robot.husky import Husky


class KitchenEnv(gym.Env):
    def __init__(
            self, omni_kit, robot_path, sparse_reward, width=320, height=240, 
            env_config_file="env_config.yml"):
        base_path = os.path.split(__file__)[0]
        path = os.path.join(base_path, env_config_file)
        with open(path, 'r') as stream:
            config = yaml.safe_load(stream)

        print(config)
        self._cube_type = config['object_type']
        self._color = config['color']
        self._size = config['size']

        self._robot_name = config['names_in_env']['robot_name']
        self._cube_name = config['names_in_env']['cube_name']
        self._place_name = config['names_in_env']['place_name']
        self._background_name = config['names_in_env']['background_name']
        self._env_name = config['name']
        self.width = width
        self.height = height

        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3])
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, self.height, self.width))

        self._robot_path = robot_path
        self.sparse_reward = sparse_reward
        if self._env_name == "on_place":
            if not config['another_place']:
                self._place_pos = [-55, 365, 89.5]
            else:
                self._place_pos = [5, 365, 89.5]
        self._cube_pos = [-26, 330, 94]
        self.cube_init_pos = self._cube_pos[:]

        if self._env_name == "on_place":
            self.initial_dist = np.linalg.norm(np.array(self._cube_pos[0:2])
                                               - np.array(self._place_pos[0:2]))
        elif self._env_name == "lift_up":
            self.prev_height = self._cube_pos[2]
        elif self._env_name == "touch_it":
            self.first_time = True


        self.omniverse_kit = omni_kit
        self.stage = self.omniverse_kit.get_stage()
        self._editor = ed.get_editor_interface()

        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        self.sd_helper = SyntheticDataHelper()
        self.cube_ar = None
        self.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)
        frame = 0
        self.dt = 1 / 60.0
        print("simulating physics...")
        while frame < 1200 or self.omniverse_kit.is_loading():
            self.omniverse_kit.update(self.dt)
            frame = frame + 1
        print("done after frame: ", frame)

    def _get_obs(self):
        gt = self.sd_helper.get_groundtruth(["rgb"])
        image = gt["rgb"][:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (self.width, self.height))
        
        return np.transpose(resized, (2, 0, 1)), np.transpose(image, (2, 0, 1))

    def get_obs(self):
        return self._get_obs()

    def step(self, action):
        self.robot.command(action)
        self.omniverse_kit.update(self.dt)
        done = self._check_if_done()[0]
        reward, finished = self._calculate_reward()
        #if np.abs(reward + 0.1) > 1e-3:
        #    print(self.robot.prev_position)
        #    print(f'non-trivial reward! {reward}')
        res, orig = self._get_obs()
        return res, reward, done or finished, {'original': orig}

    def _distance_calculation(self):
        if self._env_name == 'on_place':
            cube_rigid_body = self._dc.get_rigid_body(self._cube_name)
            cube_pos = self._dc.get_rigid_body_pose(cube_rigid_body).p
            current_pose = cube_pos[:2]
            is_fall = cube_pos[2] < 60
            distance = np.linalg.norm(current_pose - np.array(self._place_pos[:2]))
            return distance, is_fall
        elif self._env_name == 'lift_up':
            cube_rigid_body = self._dc.get_rigid_body(self._cube_name)
            cube_pos = self._dc.get_rigid_body_pose(cube_rigid_body).p
            current_pose = cube_pos[2]
            is_fall = current_pose < 80
            if current_pose >= 125:
                return 100, is_fall
            distance = current_pose - self.prev_height
            self.prev_height = current_pose
            return distance, is_fall
        elif self._env_name == 'touch_it':
            cube_rigid_body = self._dc.get_rigid_body(self._cube_name)
            manip_rigit_body = self._dc.get_rigid_body('/husky/robotiq_85_base_link')
            cube_pos = self._dc.get_rigid_body_pose(cube_rigid_body).p
            manip_pos = self._dc.get_rigid_body_pose(manip_rigit_body).p
            distance = np.linalg.norm(np.array(cube_pos) - np.array(manip_pos))
            is_touched = np.linalg.norm(cube_pos[:2] - np.array(self.cube_init_pos[:2])) > 1
            return distance, is_touched

    def _check_if_done(self):
        # place is 10x10
        if self._env_name == 'on_place':
            dist, is_fall = self._distance_calculation()
            if dist < 5 or is_fall:
                return True, is_fall
            else:
                return False, is_fall
        elif self._env_name == 'lift_up':
            dist, is_fall = self._distance_calculation()
            if dist == 100 or is_fall:
                return True, is_fall
            else:
                return False, is_fall
        elif self._env_name == 'touch_it':
            dist, is_touched = self._distance_calculation()
            if is_touched:
                return True, is_touched
            else:
                return False, is_touched

    def _calculate_reward(self):
        if self._env_name == 'on_place':
            if self.sparse_reward:
                reward = int(self._distance_calculation()[0] < 5) * 2 - 1
                return reward, reward > 0 
            else:
                reward = 0
                reward += self.initial_dist - self._distance_calculation()[0]
                self.initial_dist = self._distance_calculation()[0]
                done, is_fall = self._check_if_done()
                if reward < -10:
                    is_fall = True
                    done = True
                if done and not is_fall:
                    return 10, done
                elif done and is_fall:
                    return -10, done
            return reward, done
        elif self._env_name == 'lift_up':
            reward = 0
            reward += self._distance_calculation()[0] if self._distance_calculation()[0] > 0 else 0
            done, is_fall = self._check_if_done()
            if done and not is_fall:
                return 10, done
            elif done and is_fall:
                return -10, done
            return reward, done
        elif self._env_name == 'touch_it':
            reward = 0
            if not self.first_time:
                reward += self.initial_dist - self._distance_calculation()[0]
            else:
                self.first_time = False
            self.initial_dist = self._distance_calculation()[0]
            done, is_touched = self._check_if_done()
            if reward < -10:
                is_touched = False
                done = True
            if done and not is_touched:
                return -10, done
            elif done and is_touched:
                return 10, done
            return reward, done


    def reset(self):
        if self._env_name == 'on_place':
            self.robot.teleport(location=[-45, 260, 15], rotation=[0, 90, 0], settle=True)
            self.teleport_cube(self.cube_init_pos, [0, 90, 0])
            self._calculate_reward()
            return self._get_obs()[0]
        elif self._env_name == 'lift_up':
            self.robot.teleport(location=[-45, 260, 15], rotation=[0, 90, 0], settle=True)
            self.teleport_cube(self.cube_init_pos, [0, 90, 0])
            self._calculate_reward()
            return self._get_obs()[0]
        elif self._env_name == 'touch_it':
            self.cube_init_pos = [np.random.randint(-55, 2), np.random.randint(330, 370), 94]
            self.robot.teleport(location=[-45, 260, 15], rotation=[0, 90, 0], settle=True)
            self.teleport_cube(self.cube_init_pos, [0, 90, 0])
            self._calculate_reward()
            return self._get_obs()[0]


    def start(self):
        self._editor.set_camera_position("/OmniverseKit_Persp", 100, 340, 175, True)
        self._editor.set_camera_target("/OmniverseKit_Persp", -50, 330, 85, True)

        print("start")
        result, nucleus_server = find_nucleus_server()
        asset_path = nucleus_server + "/Users/dockeruser/Isaac"
        print(asset_path)
        print("done")

        self.setup_physics()

        # Load Environment
        self.backPrim = self.stage.DefinePrim("/World/background", "Xform")
        self.backPrim.GetReferences().AddReference( asset_path + '/Kitchen/kitchen.usd')

        # Load Robot
        self.robot = Husky(omni_kit=self.omniverse_kit, path="data/urdf/robots/husky_ur5/husky_ur5_1.urdf")
        self.robot.spawn(location=[-45, 260, 15], rotation=[0, 90, 0])

        self.cube_prim = self.stage.DefinePrim(self._cube_name, self._cube_type)
        UsdGeom.XformCommonAPI(self.cube_prim).SetTranslate(self._cube_pos)
        UsdGeom.XformCommonAPI(self.cube_prim).SetScale((self._size, self._size, self._size))
        colorAttr = UsdGeom.Gprim.Get(self.stage, self._cube_name).GetDisplayColorAttr()
        if self._color == "red":
            colorAttr.Set([(1.0, 0.0, 0.0)])
        elif self._color == "green":
            colorAttr.Set([(0.0, 1.0, 0.0)])
        elif self._color == "blue":
            colorAttr.Set([(0.0, 0.0, 1.0)])
        else:
            colorAttr.Set([(1.0, 0.0, 0.0)])
        utils.setRigidBody(self.cube_prim, "convexHull", False)

        if self._env_name == 'on_place':
            place_prim = self.stage.DefinePrim(self._place_name, "Cube")
            UsdGeom.XformCommonAPI(place_prim).SetTranslate(self._place_pos)
            UsdGeom.XformCommonAPI(place_prim).SetScale((10, 10, 1))
            colorAttr = UsdGeom.Gprim.Get(self.stage, self._place_name).GetDisplayColorAttr()
            if self._color == "red":
                colorAttr.Set([(1.0, 0.0, 0.0)])
            elif self._color == "green":
                colorAttr.Set([(0.0, 1.0, 0.0)])
            elif self._color == "blue":
                colorAttr.Set([(0.0, 0.0, 1.0)])
            else:
                colorAttr.Set([(1.0, 0.0, 0.0)])

        self.omniverse_kit.play()
        return self._get_obs()[0]

    def teleport_cube(self, pos, rot):
        handle = self._dc.get_rigid_body(self._cube_name)
        rot_quat = Gf.Rotation(Gf.Vec3d(1, 0, 0), rot).GetQuaternion()
        rot_im, rot_re = rot_quat.GetImaginary(), rot_quat.GetReal()
        tf = _dynamic_control.Transform(
            pos,
            (rot_im[0], rot_im[1], rot_im[2], rot_re),
        )
        self.omniverse_kit.update(1.0 / 60.0)
        self._dc.set_rigid_body_pose(handle, tf)
        self.omniverse_kit.update(1.0 / 60.0)

    def setup_physics(self):
        stage = self.omniverse_kit.get_stage()

        scene = PhysicsSchema.PhysicsScene.Define(stage, Sdf.Path("/World/Env/PhysicsScene"))
        # Set gravity vector
        scene.CreateGravityAttr().Set(Gf.Vec3f(0.0, 0.0, -981.0))
        # Set physics scene to use cpu physics
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/Env/PhysicsScene"))
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/Env/PhysicsScene")
        physxSceneAPI.CreatePhysxSceneEnableCCDAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableStabilizationAttr(True)
        physxSceneAPI.CreatePhysxSceneEnableGPUDynamicsAttr(False)
        physxSceneAPI.CreatePhysxSceneBroadphaseTypeAttr("MBP")
        physxSceneAPI.CreatePhysxSceneSolverTypeAttr("TGS")
        # Create physics plane for the ground
        PhysicsSchemaTools.addGroundPlane(
            stage, "/World/Env/GroundPlane", "Z", 100.0, Gf.Vec3f(0, 0, 0.1), Gf.Vec3f(1.0)
        )
        # Hide the visual geometry
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath("/World/Env/GroundPlane/geom"))
        if imageable:
            imageable.MakeInvisible()
