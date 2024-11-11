from pydrake.all import (
    Simulator,
    StartMeshcat,
    DiagramBuilder,
)
from manipulation import FindResource, running_as_notebook

from manipulation.station import (
    AddPointClouds,
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
)

from utils import *
from perception import CameraHubSystem

import os
import sys
import torch
import logging
import numpy as np
import open3d as o3d
from PIL import Image
from matplotlib import pyplot as plt

import sys
import torch

# CUDA memory management
register_signal_handlers()

### Filter out Drake warnings ###
logger = logging.getLogger('drake')
logger.addFilter(DrakeWarningFilter())

use_anygrasp = True
anygrasp_path = ""

if use_anygrasp:
    anygrasp_path = add_anygrasp_to_path()

try:
    ### Start the visualizer ###
    meshcat = StartMeshcat()
    meshcat.AddButton("Stop meshcat")

    ### Diagram setup ###
    builder = DiagramBuilder()
    scenario = LoadScenario(
        filename="robots/spot_with_arm_and_floating_base_actuators_aligned_cameras.scenario.yaml"
        # filename=FindResource(
        #     "models/spot/spot_with_arm_and_floating_base_actuators.scenario.yaml"
        # )
    )

    camera_names = []
    image_size = None

    # Enable depth images for all cameras and collect camera names (BUG: False by default?)
    for camera_name, camera_config in scenario.cameras.items():
        camera_config.depth = True
        camera_names.append(camera_name)
        
        if image_size is None:
            image_size = (camera_config.width, camera_config.height)

    ### Add objects to scene ###
    scenario = AppendDirectives(scenario, filename="objects/added_object_directives.yaml")

    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
        parser_preload_callback=lambda parser: parser.package_map().PopulateFromFolder(os.getcwd())
    ))

    ### Add pointclouds ###
    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder) # Add meshcat to parameter list to visualize PCDs (WARNING: loads but crashes after a while!)

    ### Get subsystems ###
    scene_graph = station.GetSubsystemByName("scene_graph")
    plant = station.GetSubsystemByName("plant")

    # Instantiate CameraHubSystem with station and camera_names
    camera_hub = builder.AddSystem(CameraHubSystem(station, camera_names, image_size))
    camera_hub.set_name("camera_hub")
    camera_hub.connect_cameras(station, builder)
    camera_hub.connect_point_clouds(to_point_cloud, builder)

    # Add controller = InverseDynamicsController?

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.set_name("tidyspot_diagram")
    export_diagram_as_svg(diagram, "diagram.svg")

    ### Simulation setup ###
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    ### Get subsystem contexts ###
    station_context = station.GetMyMutableContextFromRoot(context)
    plant_context = plant.GetMyMutableContextFromRoot(context)
    camera_hub_context = camera_hub.GetMyMutableContextFromRoot(context)
    # controller_context = controller.GetMyMutableContextFromRoot(context)

    ### Set initial Spot state ###
    x0 = station.GetOutputPort("spot.state_estimated").Eval(station_context)
    station.GetInputPort("spot.desired_state").FixValue(station_context, x0)

    ### Run simulation ###
    simulator.set_target_realtime_rate(1)
    simulator.set_publish_every_time_step(True)

    # meshcat.StartRecording()
    simulator.AdvanceTo(2.0)

    ################
    ### TESTZONE ###
    ################

    # Display all camera images 
    camera_hub.display_all_camera_images(camera_hub_context) 

    # Display single image
    # color_image = camera_hub.get_color_image("frontleft", camera_hub_context).data # Get color image from frontleft camera
    # plt.imshow(color_image)
    # plt.show()

    # Get the context for the camera_hub system
    camera_hub_context = camera_hub.GetMyContextFromRoot(simulator.get_context())

    # Retrieve the color and depth images from the "frontleft" camera
    color_image = camera_hub.get_color_image("frontleft", camera_hub_context).data
    depth_image = camera_hub.get_depth_image("frontleft", camera_hub_context).data

    color_image = Image.fromarray(color_image).convert("RGB")

    colors = np.array(color_image, dtype=np.float32) / 255.0
    depths = np.squeeze(depth_image)

    depths = np.squeeze(depths).copy()  # Make a writable copy
    # depths[depths == np.inf] = 0  # Replace infinite values

    # print(f"Data type of color_image: {type(colors)}")
    # print(f'Shape of color_image: {colors.shape}')

    # print(f"Data type of depth_image: {type(depths)}")
    # print(f'Shape of depth_image: {depths.shape}')

    frontleft_camera_system = station.GetSubsystemByName("rgbd_sensor_frontleft")
    camera_info = frontleft_camera_system.default_depth_render_camera().core().intrinsics()
    fx = camera_info.focal_x()
    fy = camera_info.focal_y()
    cx = camera_info.center_x()
    cy = camera_info.center_y()
    scale = 1

    # set workspace to filter output grasps
    xmin, xmax = -0.3, 0.3  # Adjusted based on your printed points range
    ymin, ymax = -0.7, 0.4  # Adjusted based on your printed points range
    zmin, zmax = 0.2, 1.0   # Adjusted to focus on the table height
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > zmin) & (points_z < zmax) & \
        (points_x > xmin) & (points_x < xmax) & \
        (points_y > ymin) & (points_y < ymax)

    points = np.stack([points_x, points_y, points_z], axis=-1)
    print(f"Points shape before mask: {points.shape}")

    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    print(f"Points shape after mask: {points.shape}")

    # import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    #################
    ### DRAKE PCD ###
    #################
    from pydrake.geometry import Rgba
    fl_context = to_point_cloud["frontleft"].GetMyContextFromRoot(simulator.get_context())
    frontleft_pcd = to_point_cloud["frontleft"].get_output_port().Eval(fl_context)

    # meshcat.SetObject("frontleft.cloud", frontleft_pcd, rgba=Rgba(1.0, 0.0, 0.0, 1.0))

    # Extract points and colors from Drake point cloud
    drake_points = frontleft_pcd.xyzs().T.astype(np.float32)          # Shape: (N, 3)
    drake_colors = frontleft_pcd.rgbs().T.astype(np.float32) / 255.0 # Shape: (N, 3)

    cam_frontleft = camera_hub._cameras["frontleft"]

    cam_frontleft_context = cam_frontleft.GetMyContextFromRoot(simulator.get_context())
    X_WC = cam_frontleft.body_pose_in_world_output_port().Eval(cam_frontleft_context)


    # print("Colors shape:", colors.shape)

    # valid_mask = np.isfinite(points).all(axis=1)

    # drake_points = drake_points[mask]
    # filtered_colors = colors[valid_mask] 

    flat_mask = mask.flatten()  # Shape: (307200,)
    drake_points = drake_points[flat_mask]

    print("drake_points shape:", drake_points.shape)

    drake_points = X_WC.inverse().multiply(drake_points.T).T

    # Check if the numpy arrays are the same
    are_arrays_equal = np.array_equal(points, drake_points)
    print("Are the numpy arrays the same?", are_arrays_equal)

    # Find the differences between the numpy arrays
    differences = points - drake_points
    print("Differences between the numpy arrays:", differences)

    # # Print the number of valid points
    # print(f"Original number of points: {points.shape[0]}")
    # print(f"Number of valid points: {filtered_points.shape[0]}")

    # # Create Open3D PointCloud object
    pcd_visual = o3d.geometry.PointCloud()
    pcd_visual.points = o3d.utility.Vector3dVector(drake_points)
    # pcd_visual.colors = o3d.utility.Vector3dVector(filtered_colors)
    # Create origin axis
    axis_length = 0.1
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])

    # Visualize the point clouds and the origin axis
    # o3d.visualization.draw_geometries([pcd_visual, pcd, axis])

    ################
    ################

    ### Anygrasp ###



    from anygrasp_sdk.grasp_detection.gsnet import AnyGrasp

    # Define configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=os.path.join(anygrasp_path, 'grasp_detection', 'checkpoints', 'checkpoint_detection.tar'), help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    cfgs = parser.parse_args(args=[])

    # Initialize AnyGrasp
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # Define workspace limits (adjust as needed)
    # lims = [-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]

    # # Define workspace limits (adjust as needed)
    # xmin, xmax = -10.0, 10.0
    # ymin, ymax = 10.0, 10.0
    # zmin, zmax = 10.0, 10.0
    # lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Run grasp detection
    gg, cloud = anygrasp.get_grasp(
        points,
        colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )

    # Check for detected grasps
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
    else:
        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]
        print(gg_pick.scores)
        print('Top grasp score:', gg_pick[0].score)

    # Visualize the detected grasps
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([pcd] + grippers)

    # # Keep meshcat alive
    # meshcat.PublishRecording()

    while not meshcat.GetButtonClicks("Stop meshcat"):
        pass

finally:
    cleanup_resources()