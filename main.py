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

from tidy_spot_planner import TidySpotPlanner
from navigation.map_helpers import PointCloudProcessor
from navigation.path_planning import DynamicPathPlanner
from controller.spot_controller import SpotController

from utils import *
from perception.ObjectDetector import ObjectDetector
from grasping.GraspSelector import GraspSelector
from grasping.PointCloudCropper import PointCloudCropper
from grasping.Grasper import Grasper

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

# Use AnyGrasp (TODO: add to args later)
use_anygrasp = False
use_grounded_sam = False

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

    # Instantiate ObjectDetector with station and camera_names
    object_detector = builder.AddSystem(ObjectDetector(station, camera_names, image_size, use_grounded_sam))
    object_detector.set_name("object_detector")
    object_detector.connect_cameras(station, builder)

    #### GRASPING ####

    # Instantiate PointCloudCropper
    point_cloud_cropper = builder.AddSystem(PointCloudCropper(camera_names))
    point_cloud_cropper.set_name("point_cloud_cropper")
    point_cloud_cropper.connect_ports(to_point_cloud, object_detector, builder)

    # Instantiate GraspSelector with use_anygrasp
    grasp_selector = builder.AddSystem(GraspSelector(use_anygrasp))
    grasp_selector.set_name("grasp_selector")
    grasp_selector.connect_ports(point_cloud_cropper, builder)

    # Instantiate Grasper
    grasper = builder.AddSystem(Grasper())
    grasper.set_name("grasper")
    grasper.connect_ports(grasp_selector, builder)

    ### PLANNER ###

    # Add point cloud processor for path planner
    point_cloud_processor = builder.AddSystem(PointCloudProcessor(station, camera_names, to_point_cloud, resolution=0.1, robot_radius=0.1))
    point_cloud_processor.set_name("point_cloud_processor")
    point_cloud_processor.connect_point_clouds(station, builder)

    # Add path planner and mapper
    dynamic_path_planner = builder.AddSystem(DynamicPathPlanner(station, point_cloud_processor, np.array([0,0,0]), resolution=0.1, robot_radius=0.1))
    dynamic_path_planner.set_name("dynamic_path_planner")
    dynamic_path_planner.connect_processor(station, builder)

    # Add controller
    controller = builder.AddSystem(SpotController(plant, use_teleop=False, meshcat=meshcat))

    # Add Finite State Machine = TidySpotPlanner
    tidy_spot_planner = builder.AddSystem(TidySpotPlanner(plant, dynamic_path_planner, controller))
    tidy_spot_planner.set_name("tidy_spot_planner")
    tidy_spot_planner.connect_components(builder, grasper, station)

    ### Build and visualize diagram ###
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
    object_detector_context = object_detector.GetMyMutableContextFromRoot(context)
    # controller_context = controller.GetMyMutableContextFromRoot(context)

    ### Set initial Spot state ###
    x0 = station.GetOutputPort("spot.state_estimated").Eval(station_context)
    station.GetInputPort("spot.desired_state").FixValue(station_context, x0)

    ### Run simulation ###
    simulator.set_target_realtime_rate(1)
    simulator.set_publish_every_time_step(True)

    meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

    # meshcat.StartRecording()
    simulator.AdvanceTo(2.0)

    ################
    ### TESTZONE ###
    ################

    # # Display all camera images
    # object_detector.display_all_camera_images(object_detector_context)

    # # Display single image
    # color_image = object_detector.get_color_image("frontleft", object_detector_context).data # Get color image from frontleft camera
    # plt.imshow(color_image)
    # plt.show()

    # # Test anygrasp on frontleft camera
    # grasp_selector.test_anygrasp_frontleft_pcd(to_point_cloud, context)

    # # Keep meshcat alive
    # meshcat.PublishRecording()

    while not meshcat.GetButtonClicks("Stop meshcat"):
        pass

finally:
    cleanup_resources()