import numpy as np
import os

from pydrake.multibody.rigid_body import RigidBody
from pydrake.all import (
        AddFlatTerrainToWorld,
        AddModelInstancesFromSdfString,
        AddModelInstanceFromUrdfFile,
        FindResourceOrThrow,
        FloatingBaseType,
        InputPort,
        Isometry3,
        OutputPort,
        RgbdCamera,
        RigidBodyPlant,
        RigidBodyTree,
        RigidBodyFrame,
        RollPitchYaw,
        RollPitchYawFloatingJoint,
        RotationMatrix,
        Value,
        VisualElement,
    )


# From
# https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices
def normalize(x):
    return x / np.linalg.norm(x)


def translate(x):
    T = np.eye(4)
    T[0:3, 3] = x[:3]
    return T


def transform_inverse(tf):
    new_tf = np.eye(4)
    new_tf[:3, :3] = tf[:3, :3].T
    new_tf[:3, 3] = -new_tf[:3, :3].dot(tf[:3, 3])
    return new_tf


def lookat(eye, target, up):
    # For a camera with +x right, +y down, and +z forward.
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    F = target[:3] - eye[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = np.cross(f, U)  # right
    u = np.cross(s, f)  # up
    M = np.eye(4)
    M[:3, :3] = np.vstack([s, -u, f]).T

    # OLD:
    # flip z -> x
    # -x -> y
    # -y -> z
    # CAMERA FORWARD is +x-axis
    # CAMERA RIGHT is -y axis
    # CAMERA UP is +z axis
    # Why does the Drake documentation lie to me???
    T = translate(eye)
    return T.dot(M)


def add_single_instance_to_rbt(rbt, config, instance_config, i):
    class_name = instance_config["class"]
    if class_name not in config["objects"].keys():
        raise ValueError("Class %s not in classes." % class_name)
    if len(instance_config["pose"]) != 6:
        raise ValueError("Class %s has pose size != 6. Use RPY plz" %
                         class_name)
    frame = RigidBodyFrame(
        "%s_%d" % (class_name, i), rbt.world(),
        instance_config["pose"][0:3],
        instance_config["pose"][3:6])
    model_path = config["objects"][class_name]["model_path"]
    _, extension = os.path.splitext(model_path)
    if extension == ".urdf":
        AddModelInstanceFromUrdfFile(
            model_path, FloatingBaseType.kFixed, frame, rbt)
    elif extension == ".sdf":
        AddModelInstancesFromSdfString(
            open(model_path), FloatingBaseType.kFixed, frame, rbt)
    else:
        raise ValueError("Class %s has non-sdf and non-urdf model name." %
                         class_name)


def setup_scene(rbt, config):
    if config["with_ground"] is True:
        AddFlatTerrainToWorld(rbt)

    for i, instance_config in enumerate(config["instances"]):
        add_single_instance_to_rbt(rbt, config, instance_config, i)

    # Add camera geometry!
    camera_link = RigidBody()
    camera_link.set_name("camera_link")
    # necessary so this last link isn't pruned by the rbt.compile() call
    camera_link.set_spatial_inertia(np.eye(6))
    camera_link.add_joint(
        rbt.world(),
        RollPitchYawFloatingJoint(
            "camera_floating_base",
            np.eye(4)))
    rbt.add_rigid_body(camera_link)

    # - Add frame for camera fixture.
    camera_frame = RigidBodyFrame(
        name="rgbd_camera_frame", body=camera_link,
        xyz=[0.0, 0., 0.], rpy=[0., 0., 0.])
    rbt.addFrame(camera_frame)
    rbt.compile()