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
        * fused_pointcloud.ply
        * scene_description.yaml
* results
    * %09d_scene
        * %s
            * estimated_scene_description.yaml
* src
    * \*.py


## Scripts in src

* `render_scene.py`: Given a scene directory with a `scene_description.yaml`,
renders depth + RGB + mask images for that scene.


## CURRENT TODOS

* Standalone utility to generate point cloud + normal for each model
* Make `render_scene` output normal maps for each image
* Make `construct_pointclouds` compile points + normals in each point cloud
* Make `construct_pointclouds` spit out point cloud files for downstream consumption
* Downstream consumption needs to happen!