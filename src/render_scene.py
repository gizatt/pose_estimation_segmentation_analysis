# -*- coding: utf8 -*-
# Given a scene directory that contains a description file
# (YAML format, see top-level README) that lists models and poses, plus a
# list of camera poses and camera calibration, renders the scene from each
# of those camera poses and saves out the resulting RGBD + per-object
# binary mask images.

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

from pydrake.multibody.rigid_body import RigidBody
from pydrake.all import (
        AddFlatTerrainToWorld,
        AddModelInstancesFromSdfString,
        AddModelInstanceFromUrdfFile,
        DiagramBuilder,
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

from utils import lookat, transform_inverse, setup_scene


def save_image_uint16(name, im):
    array_as_uint16 = im.astype(np.uint16)
    imageio.imwrite(name, array_as_uint16)


def save_image_uint8(name, im):
    array_as_uint8 = im.astype(np.uint8)
    imageio.imwrite(name, array_as_uint8)


def save_image_colormap(name, im):
    plt.imsave(name, im, cmap=plt.cm.inferno)


def save_depth_colormap(name, im, near, far):
    cmapped = plt.cm.jet((far - im)/(far - near))
    zero_range_mask = im <= near
    cmapped[:, :, 0][zero_range_mask] = 0
    cmapped[:, :, 1][zero_range_mask] = 0
    cmapped[:, :, 2][zero_range_mask] = 0
    imageio.imwrite(name, (cmapped*255.).astype(np.uint8))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("dir",
                        type=str,
                        help="Directory to find scene_description.yaml and to "
                             " save rendered images.")
    args = parser.parse_args()

    config = yaml.load(open(
        os.path.join(args.dir, "scene_description.yaml")))

    # Construct the robot and its environment
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
    context = camera.CreateDefaultContext()
    for i, viewpoint in enumerate(camera_config["viewpoints"]):
        camera_tf = lookat(viewpoint["eye"], viewpoint["target"],
                           viewpoint["up"])
        # Counteract the influence of the depth camera not being at
        # the camera frame origin
        camera_tf = camera_tf.dot(transform_inverse(depth_camera_pose))
        camera_rpy = RollPitchYaw(RotationMatrix(camera_tf[0:3, 0:3]))
        x0 = np.zeros(12)
        x0[0:3] = camera_tf[0:3, 3]
        x0[3:6] = camera_rpy.vector()
        context.FixInputPort(0, x0)
        output = camera.AllocateOutput()
        camera.CalcOutput(context, output)

        u_data = output.get_data(camera.color_image_output_port().get_index()
                                 ).get_value()
        h, w, _ = u_data.data.shape
        rgb_image = u_data.data

        save_image_uint8(
            "%s/%09d_rgb.png" % (args.dir, i), rgb_image)

        u_data = output.get_data(camera.depth_image_output_port().get_index()
                                 ).get_value()
        h, w, _ = u_data.data.shape
        depth_image = np.empty((h, w), dtype=np.float32)
        depth_image[:, :] = u_data.data[:, :, 0]
        out_of_range_mask = np.logical_or(depth_image < z_near,
                                          depth_image > z_far)
        depth_image[out_of_range_mask] = 0.

        save_depth_colormap(
            "%s/%09d_depth_color.png" % (args.dir, i),
            depth_image, z_near, z_far)
        save_image_uint8(
            "%s/%09d_depth_8bit.png" % (args.dir, i),
            255 * (depth_image-z_near)/(z_far - z_near))
        save_image_uint16(
            "%s/%09d_depth.png" % (args.dir, i),
            depth_image*1000.)

        u_data = output.get_data(camera.label_image_output_port().get_index()
                                 ).get_value()
        h, w, _ = u_data.data.shape
        label_image = np.empty((h, w), dtype=np.int8)
        label_image[:, :] = u_data.data[:, :, 0]
        # Make labels 0 where there's no label
        label_image = np.where(
            np.logical_and(label_image > 0,
                           label_image < rbt.get_num_bodies()),
            label_image,
            label_image*0)
        save_image_uint8(
            "%s/%09d_mask.png" % (args.dir, i),
            label_image)
        # Save a human-readable version for debugging
        save_image_uint8(
            "%s/%09d_mask_color.png" % (args.dir, i),
            plt.cm.jet(label_image.astype(float) / np.max(label_image))*255)

        print "Completed rendering of viewpoint %d" % i
