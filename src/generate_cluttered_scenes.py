# -*- coding: utf8 -*-

import argparse
import os
import random
import time
import yaml

import cv2
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import pydrake
from pydrake.solvers import ik
import pydrake.math as drakemath
from pydrake.multibody.rigid_body import RigidBody
from pydrake.all import (
    AbstractValue,
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    BasicVector,
    Box,
    CompliantMaterial,
    DiagramBuilder,
    Expression,
    FixedJoint,
    FloatingBaseType,
    Image,
    LeafSystem,
    PixelType,
    PortDataType,
    RgbdCamera,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RollPitchYawFloatingJoint,
    RungeKutta2Integrator,
    Shape,
    SignalLogger,
    Simulator,
    Variable,
    VisualElement
)

import transformations

from underactuated.meshcat_rigid_body_visualizer import (
    MeshcatRigidBodyVisualizer)

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g

objects = {
    "cyl_0": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_0.sdf")
        },
    "cyl_1": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_1.sdf")
        },
    "cyl_2": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_2.sdf")
        },
    "cyl_3": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_3.sdf")
        },
    "cyl_4": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_4.sdf")
        },
    "cyl_5": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_5.sdf")
        },
    "cyl_6": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_6.sdf")
        },
    "cyl_7": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_7.sdf")
        },
    "cyl_8": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_8.sdf")
        },
    "cyl_9": {
            "model_path": os.path.join(
                "models", "half_carrot_models", "cyl_9.sdf")
        },
}


def setup_scene(rbt):
    AddFlatTerrainToWorld(rbt)

    instances = []
    num_objects = np.random.randint(10)+1
    for i in range(num_objects):
        object_ind = np.random.randint(len(objects.keys()))
        pose = np.zeros(6)
        pose[0:2] = (np.random.random(2)-0.5)*0.05
        pose[2] = np.random.random()*0.1 + 0.05
        pose[3:6] = np.random.random(3)*np.pi*2.
        object_init_frame = RigidBodyFrame(
            "object_init_frame", rbt.world(),
            pose[0:3], pose[3:6])
        object_path = objects[objects.keys()[object_ind]]["model_path"]
        if object_path.split(".")[-1] == "urdf":
            AddModelInstanceFromUrdfFile(
                object_path,
                FloatingBaseType.kRollPitchYaw,
                object_init_frame, rbt)
        else:
            AddModelInstancesFromSdfString(
                open(object_path).read(),
                FloatingBaseType.kRollPitchYaw,
                object_init_frame, rbt)
        instances.append({"class": objects.keys()[object_ind],
                          "pose": pose.tolist()})
    rbt.compile()

    # Project arrangement to nonpenetration with IK
    constraints = []

    constraints.append(ik.MinDistanceConstraint(
        model=rbt, min_distance=0.01, active_bodies_idx=list(),
        active_group_names=set()))

    for body_i in range(2, 1 + num_objects):
        constraints.append(ik.WorldPositionConstraint(
            model=rbt, body=body_i, pts=np.array([0., 0., 0.]),
            lb=np.array([-0.5, -0.5, 0.0]),
            ub=np.array([0.5, 0.5, 0.5])))

    q0 = np.zeros(rbt.get_num_positions())
    options = ik.IKoptions(rbt)
    options.setDebug(True)
    options.setMajorIterationsLimit(10000)
    options.setIterationsLimit(100000)
    results = ik.InverseKin(
        rbt, q0, q0, constraints, options)

    qf = results.q_sol[0]
    info = results.info[0]
    print "Projected with info %d" % info
    return qf, instances


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for rng, "
                             "including scene generation.")
    parser.add_argument("--save",
                        action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Construct the robot and its environment
    rbt = RigidBodyTree()
    q0, instances = setup_scene(rbt)

    # Set up a visualizer for the robot
    pbrv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
    # (wait while the visualizer warms up and loads in the models)
    time.sleep(2.0)

    # Make our RBT into a plant for simulation
    rbplant = RigidBodyPlant(rbt)
    rbplant.set_name("Rigid Body Plant")
    allmaterials = CompliantMaterial()
    allmaterials.set_youngs_modulus(1E8)  # default 1E9
    allmaterials.set_dissipation(0.8)     # default 0.32
    allmaterials.set_friction(0.9)        # default 0.9.
    rbplant.set_default_compliant_material(allmaterials)
    # Build up our simulation by spawning controllers and loggers
    # and connecting them to our plant.
    builder = DiagramBuilder()
    # The diagram takes ownership of all systems
    # placed into it.
    rbplant_sys = builder.AddSystem(rbplant)
    # Hook up the visualizer we created earlier.
    visualizer = builder.AddSystem(pbrv)
    builder.Connect(rbplant_sys.state_output_port(),
                    visualizer.get_input_port(0))

    # Done!
    diagram = builder.Build()

    timestep = 0.00001
    while 1:
        # Create a simulator for it.
        simulator = Simulator(diagram)

        # The simulator simulates forward from a given Context,
        # so we adjust the simulator's initial Context to set up
        # the initial state.
        state = simulator.get_mutable_context().\
            get_mutable_continuous_state_vector()
        x0 = np.zeros(rbplant_sys.get_num_states())
        x0[0:q0.shape[0]] = q0
        state.SetFromVector(x0)

        simulator.Initialize()
        simulator.set_target_realtime_rate(1.0)
        # Simulator time steps will be very small, so don't
        # force the rest of the system to update every single time.
        simulator.set_publish_every_time_step(False)

        # From iiwa_wsg_simulation.cc:
        # When using the default RK3 integrator, the simulation stops
        # once contact happens.
        simulator.reset_integrator(
            RungeKutta2Integrator(diagram, timestep,
                                  simulator.get_mutable_context()))

        # This kicks off simulation. Most of the run time will be spent
        # in this call.
        simulator.StepTo(1.0)
        print("Final state: ", state.CopyToVector())

        if np.all(np.isfinite(state.CopyToVector())):
            break
        timestep /= 2.
        print "Halving timestep and trying again..."

    if args.save:
        export_dir = "data/scene_%09d" % (args.seed)

        # Bend over goddamn backwards to recover poses
        # since we had to spawn objects at random frames
        # to get simulation to not crash.
        kinsol = rbt.doKinematics(
            state.CopyToVector()[0:rbt.get_num_positions()])
        for i, instance in enumerate(instances):
            tf = rbt.relativeTransform(kinsol, 0, i+1)
            transformations.euler_from_matrix(tf)
            instance["pose"][0:3] = tf[:3, 3]
            instance["pose"][3:6] = transformations.euler_from_matrix(tf)
        
        config = {
            "objects": objects,
            "with_ground": True,
            "camera_config": {
                "z_near": 0.2,
                "z_far": 3.5,
                "fov_y": .7853,
                "width": 640,
                "height": 480,
                "viewpoints": [
                    {"eye": [0.5, 0., 0.25],
                     "target": [0., 0., 0.],
                     "up": [0., 0., 1.],
                     },
                    {"eye": [-0.5, 0., 0.25],
                     "target": [0., 0., 0.],
                     "up": [0., 0., 1.],
                     },
                    {"eye": [0.0, 0.5, 0.25],
                     "target": [0., 0., 0.],
                     "up": [0., 0., 1.],
                     },
                    {"eye": [0.0, -0.5, 0.25],
                     "target": [0., 0., 0.],
                     "up": [0., 0., 1.],
                     }]
            },
            "instances": instances
        }
        os.system("mkdir -p %s" % export_dir)
        with open(
                 os.path.join(export_dir, "scene_description.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

