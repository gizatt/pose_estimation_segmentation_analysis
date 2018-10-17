# -*- coding: utf8 -*-
# TODO docu

import yaml

import argparse
import itertools
import os
import parse
import random
import re
import time

import cv2
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import trimesh

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g

from icp import do_point2point_icp_fit
from cpd import do_cpd_fit

import transformations
from utils import (
    load_pointcloud, draw_points, get_pose_error,
    get_earth_movers_error)

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("method_name",
                        type=str,
                        help="[icp]")
    parser.add_argument("scene_regex",
                        type=str,
                        help="")
    parser.add_argument("-v", "--vis",
                        action='store_true',
                        help="Visualize things in meshcat?")
    parser.add_argument("-q", "--quiet",
                        action='store_true',
                        help="Don't modify files.")
    parser.add_argument("--seed",
                        default=int(time.time()),
                        type=int,
                        help="Random number seed")
    args = parser.parse_args()

    np.random.seed(args.seed % (2**31))
    random.seed(args.seed)

    vis = None
    if args.vis:
        zmq_url = "tcp://127.0.0.1:6000"
        vis = meshcat.Visualizer(zmq_url=zmq_url)

    scene_name_matcher = re.compile(args.scene_regex)

    all_method_fitter_handles = {
        "icp": do_point2point_icp_fit,
    	"cpd": do_cpd_fit
    }
    all_method_params_ranges = {
        "icp": {"n_attempts": [1],
                "max_iters_per_attempt": [100],
                "tf_init": [None],
                "model2scene": [False, True],
                "outlier_max_distance": [0.005, 0.02],
                "outlier_rejection_ratio": [0.]},
    "cpd": {"tf_init": [None]}
    }
    if args.method_name not in all_method_fitter_handles.keys():
        raise ValueError("Method %s not known!" % args.method_name)
    method_fitter_handle = all_method_fitter_handles[args.method_name]
    method_params_ranges = all_method_params_ranges[args.method_name]

    example_format = "segdist_{:f}_views_{}"
    instance_matcher = re.compile("[0-9][0-9][0-9].pc")

    do_gt_init_options = [False, True]
    earth_movers_measurement_tol = 0.005

    time_to_loop_for = 60.*60.*10  # 10  hrs!
    start_time = time.time()
    # Go through all scenes...
    try:
        while 1:
            for scene_foldername in os.listdir("data"):
                if scene_name_matcher.match(scene_foldername):
                    scene_dir = os.path.join("data", scene_foldername)
                    # Run on this scene! Load in the configuration YAML...
                    config = yaml.load(open(os.path.join(
                        scene_dir, "scene_description.yaml")))

                    # And load all model clouds
                    model_clouds_by_classname = {}

                    for class_i, class_name in enumerate(config["objects"].keys()):
                        model_clouds_by_classname[class_name] = load_pointcloud(
                            os.path.join(scene_dir, class_name + ".pc"))

                    for example_foldername in os.listdir(scene_dir):
                        parsed = parse.parse(example_format, example_foldername)
                        if parsed is not None:
                            segdist, views_string = parsed
                            views = [int(x) for x in views_string]
                            print "Segdist ", segdist, " and views...", views

                            # Open the results yaml
                            results_yaml_pathname = os.path.join(
                                "results", args.method_name, scene_foldername,
                                example_foldername)
                            os.system("mkdir -p %s" % results_yaml_pathname)
                            results_yaml_filename = os.path.join(
                                results_yaml_pathname, "results.yaml")
                            if os.path.isfile(results_yaml_filename):
                                with open(results_yaml_filename, 'r') as f:
                                    results_config = yaml.load(f)
                                    if type(results_config) != list:
                                        print "Loaded results config of " \
                                              "wrong type, nuking..."
                                        results_config = []
                            else:
                                results_config = []

                            for instance_cloud_filename in random.sample(
                                    os.listdir(os.path.join(
                                        scene_dir, example_foldername)), 1):
                                if not instance_matcher.match(
                                        instance_cloud_filename):
                                    continue
                                instance_j = int(
                                    instance_cloud_filename.split(".")[0])
                                class_name = config["instances"][instance_j][
                                    "class"]
                                gt_pose = config["instances"][instance_j]["pose"]
                                gt_tf = transformations.euler_matrix(gt_pose[3],
                                                                     gt_pose[4],
                                                                     gt_pose[5])
                                gt_tf[:3, 3] = gt_pose[:3]
                                scene_points, scene_points_normals = \
                                    load_pointcloud(os.path.join(
                                        scene_dir, example_foldername,
                                        instance_cloud_filename))

                                print "\t... loaded %d points for instance %d" % (
                                    scene_points.shape[1], instance_j)

                                if vis:
                                    print 
                                    draw_points(
                                        vis, "model", "model",
                                        scene_points,
                                        colors=plt.cm.viridis(
                                            (scene_points[2, :] - np.min(scene_points[2, :]))/
                                            (np.max(scene_points[2, :]) - np.min(scene_points[2, :]))).T
                                        )

                                # Sample params
                                method_params = {}
                                for param_name in method_params_ranges.keys():
                                    method_params[param_name] = random.choice(
                                        method_params_ranges[param_name])
                                do_gt_init = random.choice(do_gt_init_options)
                                if do_gt_init is True:
                                    method_params["tf_init"] = gt_tf.copy()

                                # Finally, run the fitter!
                                start_time = time.time()
                                est_tf = method_fitter_handle(
                                    scene_points, scene_points_normals,
                                    model_clouds_by_classname[class_name][0],
                                    model_clouds_by_classname[class_name][1],
                                    method_params, vis=vis)
                                elapsed = time.time() - start_time

                                # Compute pose TF and geodesic rotation error:
                                euclid_dist, angle_dist = get_pose_error(
                                    est_tf, gt_tf)
                                # print "\t.... Euclid dist: %f" % (euclid_dist)
                                # print "\t.... Angle dist: %f" % (angle_dist)

                                # Compute an earth-movers-distance-like error term
                                # -- in this case, the percent of model points
                                # that are more than `misalignment_tol` away from
                                # the model in its ground truth posture (using
                                # a nearest-neighbor check).
                                # Anythin over 10%>, when using
                                # misalignment_tol=1cm, indicates a bad fit.
                                earth_movers_error = get_earth_movers_error(
                                    est_tf, gt_tf,
                                    model_clouds_by_classname[class_name][0],
                                    misalignment_tol=earth_movers_measurement_tol)
                                print "\t.... Avg Earth Movers dist: %f" % (
                                    earth_movers_error)

                                results_config.append({
                                    "timestamp": time.time(),
                                    "elasped": float(elapsed),
                                    "instance": int(instance_j),
                                    "model_name": str(class_name),
                                    "views": views,
                                    "segdist": float(segdist),
                                    "gt_tf": gt_tf.tolist(),
                                    "est_tf": est_tf.tolist(),
                                    "euclid_dist": float(euclid_dist),
                                    "angle_dist": float(angle_dist),
                                    "earth_movers_error": float(
                                        earth_movers_error),
                                    "params": method_params,
                                    "do_gt_init": do_gt_init,
                                    "earth_movers_measurement_tol":
                                        earth_movers_measurement_tol
                                    })
                                break

                            if args.quiet is not True:
                                with open(results_yaml_filename, 'w') as f:
                                    yaml.dump(results_config, f,
                                              default_flow_style=False)

                            if time.time() - start_time >= time_to_loop_for:
                                print "OVERTIME"
                                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print "Interrupted"
    print "Shutting down..."
