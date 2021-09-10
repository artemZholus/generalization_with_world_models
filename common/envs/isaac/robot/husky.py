import sys
import math
import carb
import omni
import os
import numpy as np
from copy import deepcopy

from omni.isaac.urdf import _urdf
from omni.physx.scripts import physicsUtils, utils
from pxr import Usd, UsdGeom, Gf, PhysxSchema, PhysicsSchema
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.motion_planning import _motion_planning
from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server


def compute_action(end, start):
    action = deepcopy(start)
    for i in range(len(start)):
        delta = end[i] - start[i]
        delta = max(min(delta, 1), -1)
        action[i] += delta

    return action

default_position = np.array([0.0, 0.0, -1.40, -0.80, -2.40, 1.56, 0.0])

class Husky:
    def __init__(self, omni_kit, path, step_size=0.01):
        self.omni_kit = omni_kit
        self.step_size = step_size
        self.wheel_check = False

        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /Isaac folder")
            return
        self.usd_path = nucleus_server + "/Users/dockeruser/Isaac/Robots/Husky/husky.usd"

        self.robot_prim = None
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.chassis = None
        self.ar = None
        self.prev_position = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        self.default_position = np.array(default_position/step_size, dtype=np.int64)

    def spawn(self, location=(0, 0, 0), rotation=(0, 90, 0)):
        stage = self.omni_kit.get_stage()
        self.robot_prim = stage.DefinePrim("/husky", "Xform")
        self.robot_prim.GetReferences().AddReference(self.usd_path)

        physicsArticulationAPI = PhysicsSchema.ArticulationAPI.Get(stage, self.robot_prim.GetPath())
        physicsArticulationAPI.GetFixBaseAttr().Set(False)

        xform = UsdGeom.Xformable(self.robot_prim)
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
        mat = Gf.Matrix4d().SetTranslate(location)
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation))
        xform_op.Set(mat)

    def teleport(self, location, rotation, settle=True):
        if self.ar is None:
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)
            self.chassis = self.dc.get_articulation_root_body(self.ar)
        self.dc.wake_up_articulation(self.ar)

        while not np.array_equal(self.default_position, self.prev_position):
            self.prev_position = compute_action(self.default_position, self.prev_position)
            self.command(self.prev_position, is_absolute_position=True)
            self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
            self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
            self.omni_kit.update(1.0 / 60.0)

        rot_quat = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation).GetQuaternion()

        tf = _dynamic_control.Transform(
            location,
            (rot_quat.GetImaginary()[0], rot_quat.GetImaginary()[1], rot_quat.GetImaginary()[2], rot_quat.GetReal()),
        )

        self.dc.set_rigid_body_pose(self.chassis, tf)
        self.omni_kit.update(1.0 / 60.0)
        self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
        self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
        self.omni_kit.update(1.0 / 60.0)

    def command(self, vel_target, is_absolute_position=False):
        """
        Moves Husky joints as specified in vel_target array
        Args:
            vel_target: 6-d array with values in {-1, 0, 1} with
                target velocities for pan joint, lift, elbow and
                three wrist joints respectively
        """
        if not self.wheel_check:
            self.ar = self.dc.get_articulation(self.robot_prim.GetPath().pathString)

            self.pan_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_shoulder_pan_joint")
            self.lift_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_shoulder_lift_joint")
            self.elbow_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_elbow_joint")
            self.wrist1_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_wrist_1_joint")
            self.wrist2_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_wrist_2_joint")
            self.wrist3_joint = self.dc.find_articulation_dof(self.ar, "ur_arm_wrist_3_joint")
            self.left_finger1 = self.dc.find_articulation_dof(self.ar, "robotiq_85_left_finger_tip_joint")
            self.left_finger2 = self.dc.find_articulation_dof(self.ar, "robotiq_85_left_inner_knuckle_joint")
            self.left_finger3 = self.dc.find_articulation_dof(self.ar, "robotiq_85_left_knuckle_joint")
            self.right_finger1 = self.dc.find_articulation_dof(self.ar, "robotiq_85_right_finger_tip_joint")
            self.right_finger2 = self.dc.find_articulation_dof(self.ar, "robotiq_85_right_inner_knuckle_joint")
            self.right_finger3 = self.dc.find_articulation_dof(self.ar, "robotiq_85_right_inner_knuckle_joint")

            self.joints = [
                self.pan_joint,
                self.lift_joint,
                self.elbow_joint,
                self.wrist1_joint,
                self.wrist2_joint,
                self.wrist3_joint
            ]
            self.fingers = [
                self.left_finger1,
                self.left_finger2,
                self.left_finger3,
                self.right_finger1,
                self.right_finger2,
                self.right_finger3,
            ]
            self.wheel_check = True

        self.dc.wake_up_articulation(self.ar)
        if not is_absolute_position:
            self.prev_position = self.prev_position.astype(np.float32)
            self.prev_position += vel_target
            vel_target = self.prev_position
        vel_target = np.array(vel_target, dtype=np.float32)
        for f in self.fingers:
            self.dc.set_dof_position_target( f, self.step_size*vel_target[0])
            self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
            self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
        for i, joint in enumerate(self.joints):
            self.dc.set_dof_position_target(joint, self.step_size*vel_target[i + 1])
            self.dc.set_rigid_body_linear_velocity(self.chassis, [0, 0, 0])
            self.dc.set_rigid_body_angular_velocity(self.chassis, [0, 0, 0])
