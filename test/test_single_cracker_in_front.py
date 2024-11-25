import sys
import os

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

from manipulation.meshcat_utils import AddMeshcatTriad

from TidySpotFSM import TidySpotFSM
from navigation.PointCloudMapper import PointCloudMapper
from navigation.Navigator import Navigator
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

# Be sure to add the project path to PYTHONPATH before running this script
# For example, if the project is located in ~/RoboticManipulation/6.4212-TidySpot, run both:
# export PYTHONPATH=$PYTHONPATH:~/RoboticManipulation/6.4212-TidySpot
# export PYTHONPATH=$PYTHONPATH:~/RoboticManipulation/6.4212-TidySpot/third_party

# CUDA memory management
register_signal_handlers()

### Filter out Drake warnings ###
logger = logging.getLogger('drake')
logger.addFilter(DrakeWarningFilter())

# Use AnyGrasp (TODO: add to args later)
use_anygrasp = True
use_grounded_sam = True

def single_cracker_in_front_directives():
    return """
directives:
- add_model:
    name: ground_plane
    file: package://TidySpot_objects/ground_plane.sdf
- add_weld:
    parent: world
    child: ground_plane::ground_plane_center
- add_model:
    name: north_wall
    file: package://TidySpot_objects/wall.sdf
- add_weld:
    parent: ground_plane::ground_plane_center
    child: north_wall::wall_bottomcenter
    X_PC:
        translation: [4.5, 0, -0.047]
        rotation: !Rpy { deg: [0, 90, 0]}
- add_model:
    name: south_wall
    file: package://TidySpot_objects/wall.sdf
- add_weld:
    parent: ground_plane::ground_plane_center
    child: south_wall::wall_bottomcenter
    X_PC:
        translation: [-5.5, 0, -0.047]
        rotation: !Rpy { deg: [0, 90, 0]}
- add_model:
    name: east_wall
    file: package://TidySpot_objects/wall.sdf
- add_weld:
    parent: ground_plane::ground_plane_center
    child: east_wall::wall_bottomcenter
    X_PC:
        translation: [0, 4.5, -0.047]
        rotation: !Rpy { deg: [0, 90, 90]}
- add_model:
    name: west_wall
    file: package://TidySpot_objects/wall.sdf
- add_weld:
    parent: ground_plane::ground_plane_center
    child: west_wall::wall_bottomcenter
    X_PC:
        translation: [0, -5.5, -0.047]
        rotation: !Rpy { deg: [0, 90, 90]}
- add_model:
    name: cracker
    file: package://manipulation/hydro/003_cracker_box.sdf
    default_free_body_pose:
        base_link_cracker:
            translation: [1, 0, 0.5]
            rotation: !Rpy { deg: [90, 180, -90]}
"""

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
    scenario = AppendDirectives(scenario, data=single_cracker_in_front_directives())

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
    simulator.AdvanceTo(15.0)

    ################
    ### TESTZONE ###
    ################

    # # Display all camera images
    # object_detector.display_all_camera_images(object_detector_context)

    # # Display single image
    # color_image = object_detector.get_color_image("back", object_detector_context).data # Get color image from frontleft camera
    # plt.imshow(color_image)
    # plt.show()

    def display_front_images(object_detector, context):
        frontleft_image = np.rot90(object_detector.get_color_image("frontleft", context).data, k=-1)
        frontright_image = np.rot90(object_detector.get_color_image("frontright", context).data, k=-1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(frontright_image)
        axes[0].set_title("Front Right Camera")
        axes[0].axis('off')

        axes[1].imshow(frontleft_image)
        axes[1].set_title("Front Left Camera")
        axes[1].axis('off')

        plt.show()

    
    # display_front_images(object_detector, object_detector_context)

    # # Test segmentation on camera
    # object_detector.test_segmentation(object_detector_context, "back")

    # # Test cropping from segmentation on frontleft camera
    # pcd_cropper_context = point_cloud_cropper.GetMyMutableContextFromRoot(context)
    # point_cloud_cropper.test_frontleft_crop_from_segmentation(pcd_cropper_context)
    
    # # Test anygrasp on frontleft camera
    # grasp_selector.test_anygrasp_frontleft_pcd(to_point_cloud, context)

    # Test anygrasp on segmented point cloud
    # grasp_selector.test_anygrasp_frontleft_segmented_pcd(grasp_selector.GetMyMutableContextFromRoot(context))

    # Test grasp selection
    # grasp_pose = grasper.GetGraspSelection(grasper.GetMyMutableContextFromRoot(context))
    # AddMeshcatTriad(meshcat, "grasp_pose", length=0.1, radius=0.006, X_PT=grasp_pose)

    # # Keep meshcat alive
    # meshcat.PublishRecording()

    while not meshcat.GetButtonClicks("Stop meshcat"):
        pass

finally:
    cleanup_resources()