# -*- coding: utf8 -*-
# Given a scene directory with a scene description file as well
# as rendered images (using render_scene.py), generates
# point clouds for input into pose estimation algorithms.

import yaml

import argparse
import os
import random
import time

import cv2
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import trimesh

from pydrake.multibody.rigid_body import RigidBody
from pydrake.all import (
        AddFlatTerrainToWorld,
        AddModelInstancesFromSdfString,
        AddModelInstanceFromUrdfFile,
        DiagramBuilder,
        FindResourceOrThrow,
        FloatingBaseType,
        Image,
        InputPort,
        Isometry3,
        OutputPort,
        PixelType,
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

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g

from utils import (
    lookat, transform_inverse,
    add_single_instance_to_rbt, setup_scene)


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dir",
                        type=str,
                        help="Directory to find scene_description.yaml and to "
                             " save rendered images.")
    parser.add_argument("-v", "--vis",
                        action='store_true',
                        help="Visualize pointclouds in meshcat?")
    args = parser.parse_args()

    vis = None
    if args.vis:
        vis_prefix = "render_scene"
        zmq_url = "tcp://127.0.0.1:6000"
        vis = meshcat.Visualizer(zmq_url=zmq_url)
        vis[vis_prefix].delete()

    config = yaml.load(open(
        os.path.join(args.dir, "scene_description.yaml")))

    rbt = RigidBodyTree()
    setup_scene(rbt, config)

    camera_config = config["camera_config"]
    z_near = camera_config["z_near"]
    z_far = camera_config["z_far"]
    fov_y = camera_config["fov_y"]
    width = camera_config["width"]
    height = camera_config["height"]
    camera = RgbdCamera(name="camera", tree=rbt,
                        frame=rbt.findFrame("rgbd_camera_frame"),
                        z_near=z_near, z_far=z_far, fov_y=fov_y,
                        width=width, height=height, show_window=False)
    camera.set_color_camera_optical_pose(
        camera.depth_camera_optical_pose())

    depth_camera_pose = camera.depth_camera_optical_pose().matrix()

    # Build RBTs with each object individually present in them for
    # doing distance checks
    q0 = np.zeros(6)
    single_object_rbts = []
    for instance_j, instance_config in enumerate(config["instances"]):
        new_rbt = RigidBodyTree()
        add_single_instance_to_rbt(new_rbt, config,
                                   instance_config, instance_j)
        new_rbt.compile()
        single_object_rbts.append(new_rbt)

    all_points = []
    all_points_labels = []
    all_points_distances = [[] for i in range(len(config["instances"]))]

    for i, viewpoint in enumerate(camera_config["viewpoints"]):
        camera_tf = lookat(viewpoint["eye"], viewpoint["target"],
                           viewpoint["up"])
        camera_tf = camera_tf.dot(transform_inverse(depth_camera_pose))
        camera_rpy = RollPitchYaw(RotationMatrix(camera_tf[0:3, 0:3]))
        q0 = np.zeros(6)
        q0[0:3] = camera_tf[0:3, 3]
        q0[3:6] = camera_rpy.vector()
        depth_image_name = "%09d_depth.png" % (i)
        depth_image = np.array(
            imageio.imread(os.path.join(args.dir, depth_image_name))) / 1000.
        depth_image_drake = Image[PixelType.kDepth32F](
            depth_image.shape[1], depth_image.shape[0])
        depth_image_drake.mutable_data[:, :, 0] = depth_image[:, :]
        points = camera.ConvertDepthImageToPointCloud(
            depth_image_drake, camera.depth_camera_info())
        good_points_mask = np.all(np.isfinite(points), axis=0)
        points = points[:, good_points_mask]
        points = np.vstack([points, np.ones([1, points.shape[1]])])
        # Transform them to camera base frame
        points = np.dot(depth_camera_pose, points)
        # and then to world frame
        # Last body = camera floating base
        kinsol = rbt.doKinematics(q0)

        points = rbt.transformPoints(
            kinsol, points[0:3, :], rbt.get_num_bodies()-1, 0)

        label_image_name = "%09d_mask.png" % (i)
        labels = np.array(
            imageio.imread(os.path.join(args.dir, label_image_name))).ravel()
        labels = labels[good_points_mask]

        if vis:
            z_heights_normalized = (points[2, :]-np.min(points[2, :])) \
                / (np.max(points[2, :])-np.min(points[2, :]))
            label_separated_heights = z_heights_normalized + labels
            colors = cm.jet(label_separated_heights /
                            np.max(label_separated_heights)).T[0:3, :]
            vis[vis_prefix]["points_%d" % i].set_object(
                g.PointCloud(position=points,
                             color=colors,
                             size=0.005))
        all_points.append(points)
        all_points_labels.append(labels)

        # Calculate distance to each object instance
        for instance_j, instance_config in enumerate(config["instances"]):
            this_kinsol = single_object_rbts[instance_j].doKinematics(
                np.zeros(0))
            phi, _, _, _, _ = single_object_rbts[instance_j]\
                .collisionDetectFromPoints(this_kinsol,
                                           points, use_margins=False)
            all_points_distances[instance_j].append(phi)

    all_points = np.hstack(all_points)
    all_labels = np.hstack(all_points_labels)
    all_points_distances = [np.hstack(x) for x in all_points_distances]
    print "Combined to get %d points" % all_points.shape[1]

    for instance_j, instance_config in enumerate(config["instances"]):
        very_close_points = np.abs(all_points_distances[instance_j]) < 0.01
        vis[vis_prefix]["points_close_to_obj_%d" % instance_j].set_object(
                g.PointCloud(position=all_points[:, very_close_points],
                             color=None,
                             size=0.01))
