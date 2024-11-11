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
from perception.CameraHubSystem import CameraHubSystem

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
    # camera_hub.display_all_camera_images(camera_hub_context) 

    # Display single image
    # color_image = camera_hub.get_color_image("frontleft", camera_hub_context).data # Get color image from frontleft camera
    # plt.imshow(color_image)
    # plt.show()

    ##DRAKE

    frontleft_context = to_point_cloud["frontleft"].GetMyContextFromRoot(simulator.get_context())
    frontleft_pcd = to_point_cloud["frontleft"].get_output_port().Eval(frontleft_context)

    # Extract points and colors from Drake point cloud
    drake_points = frontleft_pcd.xyzs().T.astype(np.float32)          # Shape: (N, 3)
    drake_colors = frontleft_pcd.rgbs().T.astype(np.float32) / 255.0 # Shape: (N, 3)
    
    print("drake_points shape:", drake_points.shape)

    valid_mask = np.isfinite(drake_points).all(axis=1)

    drake_points = drake_points[valid_mask]
    drake_colors = drake_colors[valid_mask] 

    # # Create Open3D PointCloud object
    pcd_visual = o3d.geometry.PointCloud()
    pcd_visual.points = o3d.utility.Vector3dVector(drake_points)
    pcd_visual.colors = o3d.utility.Vector3dVector(drake_colors)
    # Create origin axis
    axis_length = 0.1
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])

    # Visualize the point clouds and the origin axis
    # o3d.visualization.draw_geometries([pcd_visual, axis])

    ################

    from anygrasp_sdk.grasp_detection.gsnet import AnyGrasp

    # Define configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default=os.path.join(anygrasp_path, 'grasp_detection', 'checkpoints', 'checkpoint_detection.tar'), help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.4, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    cfgs = parser.parse_args(args=[])

    # Initialize AnyGrasp
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # set workspace to filter output grasps
    xmin, xmax = 0, 2  # Adjusted based on your printed points range
    ymin, ymax = -1, 1  # Adjusted based on your printed points range
    zmin, zmax = 0.2, 1.0   # Adjusted to focus on the table height
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # Filter drake_points according to lims
    xmin, xmax, ymin, ymax, zmin, zmax = lims
    mask = (
        (drake_points[:, 0] >= xmin) & (drake_points[:, 0] <= xmax) &
        (drake_points[:, 1] >= ymin) & (drake_points[:, 1] <= ymax) &
        (drake_points[:, 2] >= zmin) & (drake_points[:, 2] <= zmax)
    )
    drake_points = drake_points[mask]
    drake_colors = drake_colors[mask]


    # Run grasp detection
    gg, cloud = anygrasp.get_grasp(
        drake_points,
        drake_colors,
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
    pcd.points = o3d.utility.Vector3dVector(drake_points)
    pcd.colors = o3d.utility.Vector3dVector(drake_colors)
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([pcd] + grippers)

    # # Keep meshcat alive
    # meshcat.PublishRecording()

    while not meshcat.GetButtonClicks("Stop meshcat"):
        pass

finally:
    cleanup_resources()