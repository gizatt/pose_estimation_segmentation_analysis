import math
import matplotlib.pyplot as plt
import meshcat
import numpy as np
import os
import scipy as sp
from skimage import measure
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import struct
import time
import trimesh

from pydrake.all import (
    AngleAxis,
    MathematicalProgram,
    RotationMatrix,
    RollPitchYaw,
    SolutionResult
)


from utils import transform_points, transform_inverse


def Rgba2Hex(rgb):
    ''' Turn a list of R,G,B elements (any indexable
    list of >= 3 elements will work), where each element
    is specified on range [0., 1.], into the equivalent
    24-bit value 0xRRGGBB. '''
    val = 0
    for i in range(3):
        val += (256**(2 - i)) * int(255 * rgb[i])
    return val


def return_normalized(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def get_inlier_model_and_scene_points_scene2model(
        neigh, model_points, scene_points, tf,
        outlier_max_distance=0.01, outlier_rejection_ratio=0.3):
    # Still returns results in world frame, but needs
    # the model tf to get the scene points in model
    # frame to do the closest point check.
    distances, indices = neigh.kneighbors(
        transform_points(transform_inverse(tf),
                         scene_points[0:3, :]).T,
        return_distance=True)
    inlier_mask = np.logical_or(
        distances < outlier_max_distance,
        distances < distances.mean()*outlier_rejection_ratio)[:, 0]
    if inlier_mask.sum() == 0:
        inlier_mask += True

    model_pts_inlier = model_points[:, indices[inlier_mask]][:, :, 0]
    scene_pts_inlier = scene_points[:, inlier_mask]
    return model_pts_inlier, scene_pts_inlier


def get_inlier_model_and_scene_points_model2scene(
        neigh, model_points, scene_points,
        outlier_max_distance=0.01, outlier_rejection_ratio=0.3):
    distances, indices = neigh.kneighbors(model_points[0:3, :].T,
                                          return_distance=True)
    inlier_mask = np.logical_or(
        distances < outlier_max_distance,
        distances < distances.mean()*outlier_rejection_ratio)[:, 0]
    if inlier_mask.sum() == 0:
        inlier_mask += True

    model_pts_inlier = model_points[:, inlier_mask]
    scene_pts_inlier = scene_points[:, indices[inlier_mask]][:, :, 0]
    return model_pts_inlier, scene_pts_inlier


def draw_model_fit_pts(vis_handle, model_points, tf, color):
    color = np.array(color)
    if len(color.shape) == 1 or color.shape[1] == 1:
        colors = np.tile(
            color, [model_points.shape[1], 1]).T
    else:
        colors = color

    vis_handle.set_object(
      meshcat.geometry.PointCloud(
        position=model_points[0:3, :],
        color=colors))
    vis_handle.set_transform(tf)


def do_point2point_icp_fit(scene_points, scene_points_normals,
                           model_points, model_points_normals,
                           params, vis=None):
    n_attempts = params["n_attempts"]
    max_iters_per_attempt = params["max_iters_per_attempt"]
    model2scene = params["model2scene"]
    outlier_rejection_ratio = params["outlier_rejection_ratio"]
    outlier_max_distance = params["outlier_max_distance"]
    if vis:
        vis = vis["point2point_icp"]
    scores = []
    tfs = []
    scales = []
    tf_init = params["tf_init"]

    if vis:
        vis["fits"].delete()
    if tf_init is not None and n_attempts != 1:
        print "Setting n_attempts to 1, as you supplied"
        print " an initial tf and this algorithm should be"
        print " deterministic."
        n_attempts = 1

    for k in range(n_attempts):
        # Init transform
        if tf_init is None:
            tf = np.eye(4)
            tf[0:3, 3] = scene_points.mean(axis=1) + \
                np.random.randn(3)*np.std(scene_points, axis=1)
            tf[0:3, 0:3] = RotationMatrix(
                RollPitchYaw(np.random.random(3)*2*np.pi)).matrix()
        else:
            tf = tf_init
        scale = np.eye(4)

        neigh = NearestNeighbors(n_neighbors=1)
        if model2scene:
            neigh.fit(scene_points.T)
        else:
            neigh.fit(model_points.T)

        if vis:
            obj_name = "fits/obj_%d" % k
            draw_model_fit_pts(
                vis[obj_name], model_points, tf, [1., 0.8, 0.])

        num_steps_at_standstill = 0
        for i in range(max_iters_per_attempt):
            if vis:
                vis[obj_name].set_transform(tf)

            # Compute rest with new scaled model points
            tf_model_points = transform_points(tf, model_points)
            # (Re)compute nearest neighbors for some of these methods

            if model2scene:
                model_pts_inlier, scene_pts_inlier = \
                    get_inlier_model_and_scene_points_model2scene(
                        neigh, tf_model_points, scene_points,
                        outlier_max_distance, outlier_rejection_ratio)
            else:
                model_pts_inlier, scene_pts_inlier = \
                    get_inlier_model_and_scene_points_scene2model(
                        neigh, tf_model_points, scene_points, tf,
                        outlier_max_distance, outlier_rejection_ratio)
            if vis:
                # Vis correspondences
                n_pts = scene_pts_inlier.shape[1]

                pt = np.zeros(3, N)
                tf = 4x4


                lines = np.zeros([3, n_pts*2])
                colors = np.zeros([4, n_pts*2]).T
                inds = np.array(range(0, n_pts*2, 2))
                lines[:, inds] = model_pts_inlier[0:3, :]
                lines[:, inds+1] = scene_pts_inlier[0:3, :]
                vis["corresp"].set_object(
                    meshcat.geometry.LineSegmentsGeometry(
                        lines, colors))

            # Point-to-point ICP update with trimesh
            new_tf, _, _ = trimesh.registration.procrustes(
                model_pts_inlier[0:3, :].T,
                scene_pts_inlier[0:3, :].T,
                reflection=False,
                translation=True,
                scale=False)

            tf = new_tf.dot(tf)
            if np.allclose(np.diag(new_tf[0:3, 0:3]), [1., 1., 1.]):
                angle_dist = 0.
            else:
                # Angle from rotation matrix
                angle_dist = np.arccos(
                    (np.sum(np.diag(new_tf[0:3, 0:3])) - 1) / 2.)
            euclid_dist = np.linalg.norm(new_tf[0:3, 3])
            if euclid_dist < 0.0001 and angle_dist < 0.0001:
                num_steps_at_standstill += 1
                if num_steps_at_standstill > 10:
                    break
            else:
                num_steps_at_standstill = 0

        # Compute final fit cost
        tf_model_points = transform_points(tf, model_points)
        if model2scene:
            model_pts_inlier, scene_pts_inlier = \
                get_inlier_model_and_scene_points_model2scene(
                    neigh, tf_model_points, scene_points,
                    outlier_max_distance, outlier_rejection_ratio)
        else:
            model_pts_inlier, scene_pts_inlier = \
                get_inlier_model_and_scene_points_scene2model(
                    neigh, tf_model_points, scene_points, tf,
                    outlier_max_distance, outlier_rejection_ratio)
        score = np.square((model_pts_inlier - scene_pts_inlier)).mean()
        tfs.append(tf)
        scores.append(score)
        scales.append(scale)
        if vis:
            draw_model_fit_pts(
                vis[obj_name], model_points, tf,
                plt.cm.jet(score)[0:3])
        # print "%d: Mean resulting surface SDF value: %f" % (k, score)
        time.sleep(1.0)
        
    best_run = np.argmin(scores)
    # print "Best run was run %d with score %f" % (best_run, scores[best_run])
    return tfs[best_run]
