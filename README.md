## Directory Structure

Each scene may have more than one image (with pose information in info_%09d.yaml), fused into a common pointcloud in world frame
in fused_pointcloud.ply.

Layout:
* models
    * %s.obj
* data
    * %09d_scene
        * %09d_rgb.png
        * %09d_depth.png
        * %02d_%09d_mask.png
        * %09d_info.yaml
        * segdist_%02.03f_views_%b%b%b%b... (0 or 1 per camera view to indicate inclusion)
            * %03d.pts (Pointcloud cropped around object by index)
        * scene_description.yaml
        * model_name.pc (generated point cloud + normals for each object class name)

* results
    * %s (Technique name)
        * %09d_scene
            * segdist_%02.03f_views_%b%b%b%b... (0 or 1 per camera view to indicate inclusion)
                * fits.yaml (Contains *list* of fit attempts with instance, error, etc annotations inline)
* src
    * \*.py


## Scripts in src

* `render_scene.py`: Given a scene directory with a `scene_description.yaml`,
renders depth + RGB + mask images for that scene.
* `construct_pointclouds.py`: Given a scene directory with a
`scene_description.yaml`, fuses the depth images from that directory to produce
synthetic reconstructed point clouds with normals. Saves out
point clouds for each of the conditions we plan on testing: variation over
segmentation quality (by nearest neighbor distance -- TODO, maybe
also try by pixel distance in depth images?) and number of views in the
reconstruction.


## Questions to answer (in rough order)

* Under different levels of model occlusion, is the
ground truth pose close to a fixed point? (Given perfect scene segmentation
of scenes from 1, 2, and 3 cameras, does each technique return the ground
truth pose when seeded from the ground truth pose?)

* How does technique accuracy covary with different levels of model occlusion
and segmentation quality?


## CURRENT TODOS

* Make a generic downstream consumer interface that loads in model + scene pointcloud + normals,
  queries the downstream technique for an estimated pose, and scores it (record timing, record
  euclidean + rotational distance, and also record visual surface discrepancy).
* Implement above interface for vanilla (parameterized outlier-rejection) ICP