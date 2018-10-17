import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
import parse
import pickle
import ast
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_regex",
                        type=str,
                        default="scene_0.*",
                        help="")
    parser.add_argument("--use_cache",
                        action='store_true')
    args = parser.parse_args()

    scene_name_matcher = re.compile(args.scene_regex)

    example_format = "segdist_{:f}_views_{}"
    instance_matcher = re.compile("[0-9][0-9][0-9].pc")

    earth_movers_failure_tol = 0.1

    if args.use_cache is False:
        collated_data = {}
        for method_name in os.listdir("results"):
            if not os.path.isdir(os.path.join("results", method_name)):
                continue
            print "Collecting trials for method %s..." % method_name
            collated_data[method_name] = []
            for scene_foldername in os.listdir(
                    os.path.join("results", method_name)):
                if scene_name_matcher.match(scene_foldername):
                    scene_dir = os.path.join("data", scene_foldername)
                    # Load in the real scene configuration YAML...
                    config = yaml.load(open(os.path.join(
                        scene_dir, "scene_description.yaml")))

                    # And iterate across all conditions
                    for example_foldername in os.listdir(scene_dir):
                        parsed = parse.parse(example_format, example_foldername)
                        if parsed is not None:
                            segdist, views_string = parsed

                            # Open the results yaml
                            results_yaml_pathname = os.path.join(
                                "results", method_name, scene_foldername,
                                example_foldername)
                            results_yaml_filename = os.path.join(
                                results_yaml_pathname, "results.yaml")
                            if os.path.isfile(results_yaml_filename):
                                with open(results_yaml_filename, 'r') as f:
                                    results_config = yaml.load(f)
                            else:
                                continue

                            [collated_data[method_name].append(result)
                             for result in results_config]

            print "For method %s, collected %d results" % (
                method_name, len(collated_data[method_name]))

        with open("results/analysis_cache.pickle", "wb") as f:
            pickle.dump(collated_data, f)
    else:
        with open("results/analysis_cache.pickle", "rb") as f:
            collated_data = pickle.load(f)

    # Under different levels of model occlusion but perfect scene
    # segmentation, is the ground truth pose close to a fixed point?
    # (Given perfect scene segmentation of scenes from 1, 2, and 3 cameras,
    # does each technique return the ground truth pose when seeded from
    # the ground truth pose?)
    results_by_condition = {}
    for method_name in ["icp"]:
        samples_by_condition = {}
        for result in collated_data[method_name]:
            if result["do_gt_init"] is True:
                n_views = np.array(result["views"]).sum()
                segdist = result["segdist"]
                if segdist != 0.005:
                    continue

                correct = result["earth_movers_error"] < \
                    earth_movers_failure_tol
                model2scene = result["params"]["model2scene"]
                outlier_max_dist = result["params"]["outlier_max_distance"]    # np.arange(0., 0.05, 0.005),
                outlier_rejection_ratio = result["params"]["outlier_rejection_ratio"]    # np.arange(0., 0.5, 0.05)}

                # Cases we case about:
                if model2scene is True:
                    condition_name = "model2scene_"
                else:
                    condition_name = "scene2model_"
                if outlier_max_dist in [0.005, 0.02]:
                    condition_name += "maxdist%0.03f" % outlier_max_dist
                else:
                    continue
                if outlier_rejection_ratio != 0.0:
                    continue
                if condition_name not in samples_by_condition.keys():
                    samples_by_condition[condition_name] = []
                samples_by_condition[condition_name].append([n_views, correct])

        for condition_name in samples_by_condition.keys():
            samples = np.vstack(samples_by_condition[condition_name]).T
            results = []
            for n_views in np.unique(samples[0, :]):
                mask = samples[0, :] == n_views
                vals = samples[1, mask]
                success_rate = vals.mean()
                trials = mask.sum()
                # *Population* standard error of mean
                success_sem = vals.std(ddof=1)/np.sqrt(trials)
                results.append([n_views, success_rate, success_sem, trials])
            results_by_condition[condition_name] = np.vstack(results).T
    print results_by_condition

    for outer in ["scene2model", "model2scene"]:
        plt.figure()
        ax = plt.subplot(1, 1, 1)
        for i, condition_name in enumerate(results_by_condition.keys()):
            if condition_name.split("_")[0] == outer:
                results = results_by_condition[condition_name]
                plt.errorbar(results[0, :], results[1, :],
                             yerr=results[2, :], label=condition_name,
                             capsize=4, capthick=2, fmt='o-')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.8])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=2)
        plt.xlabel("Number of views of object")
        plt.ylabel("Pose estimate success rate")
        plt.title("Success rates when seeded at ground truth position\n"
                  "for near-perfect segmentation quality (5mm).")
        plt.ylim([-0.1, 1.2])
        plt.xticks([1, 2, 3, 4])
        plt.grid(True)

    #plt.show()

    # Similar plot but for ground truth seeding across segmentation
    # qualities
    results_by_condition = {}
    for method_name in ["icp"]:
        samples_by_condition = {}
        for result in collated_data[method_name]:
            if result["do_gt_init"] is True:
                n_views = np.array(result["views"]).sum()
                segdist = result["segdist"]
                correct = result["earth_movers_error"] < \
                    earth_movers_failure_tol
                model2scene = result["params"]["model2scene"]
                outlier_max_dist = result["params"]["outlier_max_distance"]    # np.arange(0., 0.05, 0.005),
                outlier_rejection_ratio = result["params"]["outlier_rejection_ratio"]    # np.arange(0., 0.5, 0.05)}

                # Cases we case about:
                if model2scene is True:
                    condition_name = "model2scene_"
                else:
                    condition_name = "scene2model_"
                if outlier_max_dist in [0.005, 0.02]:
                    condition_name += "maxdist%0.03f" % outlier_max_dist
                else:
                    continue
		condition_name += "_nviews%d" % n_views
                if outlier_rejection_ratio != 0.0:
                    continue
                if condition_name not in samples_by_condition.keys():
                    samples_by_condition[condition_name] = []
                samples_by_condition[condition_name].append([segdist, correct])

        for condition_name in samples_by_condition.keys():
            samples = np.vstack(samples_by_condition[condition_name]).T
	    segdists = samples[0, :]
	    vals = samples[1, :]
	    segdists_unique = np.unique(segdists)
	    means = []
	    success_sems = []
	    for segdist in segdists_unique:
		vals_here = vals[segdists == segdist]
		means.append(vals_here.mean())
		success_sems.append(vals_here.std(ddof=1)/np.sqrt(np.sum(segdists == segdist)))

	    results_by_condition[condition_name] = [segdists_unique, means, success_sems, segdists, vals]
    print results_by_condition
	
    for i, w2w in enumerate(["scene2model", "model2scene"]):
        plt.figure()
	for j, inner in enumerate(["maxdist0.005", "maxdist0.020"]):
        	ax = plt.subplot(1, 2, j+1)
		k = 0
		segdists_all = []
		for condition_name in sorted(results_by_condition.keys()):
		    if w2w in condition_name and inner in condition_name:
		        results = results_by_condition[condition_name]
			segdists_all.append(results[0]*100)
		        plt.errorbar(results[0]*100, results[1],
		                     yerr=results[2], label=condition_name,
		                     capsize=4, capthick=2, fmt='o-')
			k += 1
		if k == 0:
			print "Found no conditios for ", w2w, " / ", inner
		segdists_all = np.unique(np.hstack(segdists_all))
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1,
		                 box.width, box.height * 0.8])
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
		          fancybox=True, shadow=True, ncol=2)
		plt.xlabel("Segmentation distance (cm)")
		plt.ylabel("Pose estimate success rate")
		plt.ylim([-0.1, 1.2])

	        plt.xticks(segdists_all)
		plt.xlim([0., max(segdists_all) + 1])
		plt.title("Success rates when seeded at ground truth position\n"
		  "across segmentation qualitys for %s, maxdist=%s" % (w2w, inner))
		plt.grid(True)


    # Under different levels of scene segmentation with constant
    # model occlusion, how well do things work?
    results_by_condition = {}
    for method_name in ["icp"]:
        samples_by_condition = {}
        for result in collated_data[method_name]:
            if result["do_gt_init"] is False:
                n_views = np.array(result["views"]).sum()
                segdist = result["segdist"]
                correct = result["earth_movers_error"] < \
                    earth_movers_failure_tol
                model2scene = result["params"]["model2scene"]
                outlier_max_dist = result["params"]["outlier_max_distance"]    # np.arange(0., 0.05, 0.005),
                outlier_rejection_ratio = result["params"]["outlier_rejection_ratio"]    # np.arange(0., 0.5, 0.05)}

                # Cases we case about:
                if model2scene is True:
                    condition_name = "model2scene_"
                else:
                    condition_name = "scene2model_"
                if outlier_max_dist in [0.005, 0.02]:
                    condition_name += "maxdist%0.03f" % outlier_max_dist
                else:
                    continue
		if n_views == 1:
			condition_name += "_occluded"
		else:
			condition_name += "_unoccluded"
                if outlier_rejection_ratio != 0.0:
                    continue
                if condition_name not in samples_by_condition.keys():
                    samples_by_condition[condition_name] = []
                samples_by_condition[condition_name].append([segdist, correct])

        for condition_name in samples_by_condition.keys():
            samples = np.vstack(samples_by_condition[condition_name]).T
	    segdists = samples[0, :]
	    vals = samples[1, :]
	    segdists_unique = np.unique(segdists)
	    means = []
	    success_sems = []
	    for segdist in segdists_unique:
		vals_here = vals[segdists == segdist]
		means.append(vals_here.mean())
		success_sems.append(vals_here.std(ddof=1)/np.sqrt(np.sum(segdists == segdist)))

	    results_by_condition[condition_name] = [segdists_unique, means, success_sems, segdists, vals]
    print results_by_condition

    plt.figure()
    for i, w2w in enumerate(["scene2model", "model2scene"]):
	for j, inner in enumerate(["maxdist0.005", "maxdist0.020"]):
        	ax = plt.subplot(2, 2, i*2+j+1)
		k = 0
		segdists_all = []
		for condition_name in sorted(results_by_condition.keys()):
		    if w2w in condition_name and inner in condition_name:
		        results = results_by_condition[condition_name]
			segdists_all.append(results[0]*100)
		        plt.errorbar(results[0]*100, results[1],
		                     yerr=results[2], label=condition_name,
		                     capsize=4, capthick=2, fmt='o-')
			k += 1
		if k == 0:
			print "Found no conditios for ", w2w, " / ", inner
		segdists_all = np.unique(np.hstack(segdists_all))
		box = ax.get_position()
		ax.set_position([box.x0, box.y0 + box.height * 0.1,
		                 box.width, box.height * 0.8])
		ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
		          fancybox=True, shadow=True, ncol=2)
		plt.xlabel("Segmentation distance (cm)")
		plt.ylabel("Pose estimate success rate")
		plt.ylim([-0.1, 1.2])
		plt.title("Success rates when seeded randomly\n"
		  "for %s, maxdist = %s" % (w2w, inner))

	        plt.xticks(segdists_all)
		plt.xlim([0., max(segdists_all) + 1])
		plt.grid(True)
    plt.show()
