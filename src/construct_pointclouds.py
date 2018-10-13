# -*- coding: utf8 -*-
# Given a scene directory with a scene description file as well
# as rendered images (using render_scene.py), generates
# point clouds for input into pose estimation algorithms.
# Generates both scene clouds (at different levels of segmentation)
# and model clouds for each model.

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


def sample_points_on_body(rbt, body_index, density):
    all_points = []
    all_normals = []
    body = rbt.get_body(body_index)
    for visual_element in body.get_visual_elements():
        if visual_element.hasGeometry():
            geom = visual_element.getGeometry()
            tf = visual_element.getLocalTransform()
            points = geom.getPoints()
            if geom.hasFaces():
                faces = geom.getFaces()
                new_points_pretf = []
                new_normals_pretf = []
                for face in faces:
                    v1 = points[:, face[0]]
                    v2 = points[:, face[1]]
                    v3 = points[:, face[2]]
                    # Roughly grid-like iteration over the triangle
                    # surface -- not precise.
                    # For each step along v1 --> v3,
                    # step across the tri in the v1 --> v2 direction.
                    v1_v2_dist = np.linalg.norm(v2 - v1)
                    v1_v3_dist = np.linalg.norm(v3 - v1)
                    v1_v2_n = (v2 - v1) / v1_v2_dist
                    v1_v3_n = (v3 - v1) / v1_v3_dist
                    n_steps = 0
                    for outer_step in np.arange(0., v1_v3_dist, density):
                        # We've advanced some percent up the triangle from
                        # v1->v3, which means the distance *across* from
                        # v1->v2 is less by a similar ratio.
                        ratio = 1. - outer_step / v1_v3_dist
                        for inner_step in np.arange(0., v1_v2_dist*ratio,
                                                    density):
                            pt = v1 + outer_step*v1_v3_n + inner_step*v1_v2_n
                            new_points_pretf.append(pt)
                            n_steps += 1

                    # Normal assumes vertices listed in ccw order
                    normal = np.cross(v1_v2_n, v1_v3_n)
                    new_normals_pretf.append(np.tile(normal, [n_steps, 1]))

                new_points_pretf = np.vstack(new_points_pretf).T
                new_normals_pretf = np.vstack(new_normals_pretf).T
            else:
                raise ValueError("No-faces case not implemented...")
            new_points = ((tf[:3, :3].dot(new_points_pretf)).T + tf[:3, 3]).T
            new_normals = tf[:3, :3].dot(new_normals_pretf)
            all_points.append(new_points)
            all_normals.append(new_normals)
    return np.hstack(all_points), np.hstack(all_normals)


def do_model_pointcloud_sampling(args, config, vis=None):
    # For each class, sample model points on its surface.
    for class_i, class_name in enumerate(config["objects"].keys()):
        class_rbt = RigidBodyTree(config["objects"][class_name]["model_path"])
        # Sample model points
        model_points, model_normals = sample_points_on_body(class_rbt, 1, 0.005)
        print "Sampled %d model points from model %s" % (
            model_points.shape[1], class_name)
        save_pointcloud(model_points, model_normals,
                        os.path.join(args.dir, "%s.pc" % (class_name)))
        if vis:
            model_pts_offset = (model_points.T + [class_i, 0., -1.0]).T
            vis[vis_prefix]["points_%s" % class_name].set_object(
                g.PointCloud(position=model_pts_offset,
                             color=None,
                             size=0.001))
            n_pts = model_points.shape[1]
            # Drawing normals for debug
            lines = np.zeros([3, n_pts*2])
            inds = np.array(range(0, n_pts*2, 2))
            lines[:, inds] = model_pts_offset[0:3, :]
            lines[:, inds+1] = model_pts_offset[0:3, :] + model_normals * 0.01
            vis[vis_prefix]["normals_%s" % class_name].set_object(
                meshcat.geometry.LineSegmentsGeometry(
                    lines, plt.cm.jet(np.arange(0., 1., n_pts*2))))


def save_pointcloud(pc, normals, path):
    joined = np.hstack([pc.T, normals.T])
    np.savetxt(path, joined)


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

    do_model_pointcloud_sampling(args, config, vis)

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
    # doing distance checks, and for generating model point clouds
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
