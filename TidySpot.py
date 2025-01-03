from pydrake.all import (
    Simulator,
    StartMeshcat,
    DiagramBuilder,
    StateInterpolatorWithDiscreteDerivative,
)
from manipulation import FindResource, running_as_notebook

from manipulation.station import (
    AppendDirectives,
    LoadScenario,
    # MakeHardwareStation,
)

from TidySpotFSM import TidySpotFSM
from navigation.PointCloudMapper import PointCloudMapper
from navigation.Navigator import Navigator

from controller.PositionCombiner import PositionCombiner
from controller.SpotArmIKController import SpotArmIKController

from utils.utils import *
# from station import MakeHardwareStation
from utils.tidyspotHardwareStation import MakeHardwareStation

from perception.ObjectDetector import ObjectDetector
from grasping.GraspSelector import GraspSelector
from grasping.PointCloudCropper import PointCloudCropper

from manipulation.meshcat_utils import AddMeshcatTriad

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

def run_TidySpot(args):
    # CUDA memory management
    register_signal_handlers()
    ### Filter out Drake warnings ###
    logger = logging.getLogger('drake')
    logger.addFilter(DrakeWarningFilter())


    use_anygrasp = args.grasp_type == "anygrasp"
    use_grounded_sam = args.perception_type == "sam"
    device = args.device
    scenario_path = args.scenario
    automatic_clutter_generation = False

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
            if camera_name != "hand" and camera_name != "back":
                camera_names.append(camera_name)

            if image_size is None:
                image_size = (camera_config.width, camera_config.height)

        ### Add objects to scene ###
        scenario = AppendDirectives(scenario, filename=scenario_path)
        if automatic_clutter_generation:
            forbidden_areas = [((1.5, 1.5), (-1.5, -1.5))]
            clutter_path = clutter_gen(3, (3.0, 3.0), "objects/clutter.yaml", forbidden_areas)
            scenario = AppendDirectives(scenario, filename=clutter_path)

        # Get bin location from scenario (assuming bin link is welded to world)
        bin_location = get_bin_translation(scenario, bin_link_name="planar_bin::bin_base")
        print(f"Bin location: {bin_location}")

        station = builder.AddSystem(MakeHardwareStation(
            scenario=scenario,
            meshcat=meshcat,
            # parser_preload_callback=lambda parser: parser.package_map().AddPackageXml("robots/spot_description/package.xml")
            parser_preload_callback=lambda parser: parser.package_map().PopulateFromFolder(os.getcwd())
        ))

        ### Add pointclouds ###
        to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder) # Add meshcat to parameter list to visualize PCDs (WARNING: loads but crashes after a while!)

        ### Get subsystems ###
        scene_graph = station.GetSubsystemByName("scene_graph")
        plant = station.GetSubsystemByName("plant")

        # Instantiate ObjectDetector with station and camera_names
        object_detector = builder.AddSystem(ObjectDetector(station, camera_names, image_size, use_grounded_sam, device=device))
        object_detector.set_name("object_detector")
        object_detector.connect_cameras(station, builder)

        #### GRASPING ####

        # Instantiate PointCloudCropper
        point_cloud_cropper = builder.AddSystem(PointCloudCropper(camera_names, meshcat=meshcat))
        point_cloud_cropper.set_name("point_cloud_cropper")
        point_cloud_cropper.connect_ports(to_point_cloud, object_detector, builder)

        # Instantiate GraspSelector with use_anygrasp
        grasp_selector = builder.AddSystem(GraspSelector(use_anygrasp, plant=plant, scene_graph=scene_graph, meshcat=meshcat))
        grasp_selector.set_name("grasp_selector")
        grasp_selector.connect_ports(station, point_cloud_cropper, builder)

        ### PLANNER ###

        # Add point cloud mapper for path planner
        bin_size = (0.8, 0.8) # HARDCODED BIN SIZE
        point_cloud_mapper = builder.AddSystem(PointCloudMapper(station, camera_names, to_point_cloud, bin_location=bin_location, bin_size=bin_size, resolution=0.1,  meshcat=meshcat))
        point_cloud_mapper.set_name("point_cloud_mapper")
        point_cloud_mapper.connect_components(point_cloud_cropper, station, builder)

        # Add path planner and mapper
        navigator = builder.AddSystem(Navigator(station, builder, np.array([0,0,0]), resolution=0.1, bin_location=bin_location, meshcat=meshcat, visualize=False))
        navigator.set_name("navigator")
        navigator.connect_mapper(point_cloud_mapper, station, builder)

        ### CONTROLLING ###

        # Add IK controller for to solve arm positions for grasping
        spot_plant = station.GetSubsystemByName("spot.controller").get_multibody_plant_for_control()
        spot_arm_ik_controller = builder.AddSystem(SpotArmIKController(plant, spot_plant, bin_location, meshcat=meshcat, use_anygrasp=use_anygrasp))
        spot_arm_ik_controller.set_name("spot_arm_ik_controller")
        spot_arm_ik_controller.connect_components(builder, station, navigator, grasp_selector)

        # Add position combiner to combine base and arm position commands
        position_combiner = builder.AddSystem(PositionCombiner())
        position_combiner.set_name("position_combiner")
        position_combiner.connect_components(builder, spot_arm_ik_controller, navigator)

        ### FSM ###

        # Add Finite State Machine = TidySpotFSM
        tidy_spot_planner = builder.AddSystem(TidySpotFSM(plant, bin_location))
        tidy_spot_planner.set_name("tidy_spot_fsm")
        tidy_spot_planner.connect_components(builder, object_detector, grasp_selector, spot_arm_ik_controller, point_cloud_mapper, navigator, station)

        # Last component, add state interpolator which converts desired state to desired state and velocity
        state_interpolator = builder.AddSystem(StateInterpolatorWithDiscreteDerivative(10, 0.1, suppress_initial_transient=True))
        state_interpolator.set_name("state_interpolator")

        # Connect desired state through interpolator to robot
        builder.Connect(
            position_combiner.GetOutputPort("spot_commanded_state"),
            state_interpolator.get_input_port()
        )
        builder.Connect(
            state_interpolator.get_output_port(),
            station.GetInputPort("spot.desired_state")
        )

        ### Build and visualize diagram ###
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.set_name("tidyspot_diagram")
        export_diagram_as_svg(diagram, "diagram.svg")
        grasp_selector.set_diagram(diagram)

        ### Simulation setup ###
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()

        ### Get subsystem contexts ###
        station_context = station.GetMyMutableContextFromRoot(context)
        plant_context = plant.GetMyMutableContextFromRoot(context)
        object_detector_context = object_detector.GetMyMutableContextFromRoot(context)
        # controller_context = controller.GetMyMutableContextFromRoot(context)

        inverse_dynamics_controller = station.GetSubsystemByName("spot.controller")

        ### Set initial Spot state ###
        x0 = station.GetOutputPort("spot.state_estimated").Eval(station_context)
        # Don't FixValue here - this will override the controller's commands
        # station.GetInputPort("spot.desired_state").FixValue(station_context, x0)


        ### Run simulation ###
        simulator.set_target_realtime_rate(1)
        simulator.set_publish_every_time_step(True)

        # Print states at each time step
        def PrintStates(context):
            # Get subsystem contexts
            station_context = station.GetMyContextFromRoot(context)
            path_planner_context = navigator.GetMyContextFromRoot(context)

            # Evaluate ports with correct contexts
            desired_state = station.GetInputPort("spot.desired_state").Eval(station_context)
            actual_state = station.GetOutputPort("spot.state_estimated").Eval(station_context)
            path_planner_state = navigator.GetOutputPort("desired_state").Eval(path_planner_context)

            print(f"\nTime: {context.get_time():.2f}")
            print(f"Desired state (to robot): {desired_state}")  # Just print base position for clarity
            print(f"Path planner output: {path_planner_state}")
            print(f"Actual state: {actual_state}")

            # Also print path planner status
            execute_path = navigator.get_input_port(navigator._execute_path_input_index).Eval(path_planner_context)
            goal = navigator.get_input_port(navigator._goal_input_index).Eval(path_planner_context)
            print(f"Execute path: {bool(execute_path[0])}")
            print(f"Goal: {goal[:3]}")
            print("---")

        q9_pid_output_history = []
        times = []

        def visualize_controller_poses_and_debug(context):
            gripper_pose = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("arm_link_fngr"))
            AddMeshcatTriad(meshcat, "arm_link_fngr", length=0.1, radius=0.006, X_PT=gripper_pose)
            if spot_arm_ik_controller.prepick_pose is not None:
                AddMeshcatTriad(meshcat, "prepick_pose", length=0.1, radius=0.006, X_PT=spot_arm_ik_controller.prepick_pose)
            if spot_arm_ik_controller.desired_gripper_pose is not None:
                AddMeshcatTriad(meshcat, "desired_gripper_pose", length=0.1, radius=0.006, X_PT=spot_arm_ik_controller.desired_gripper_pose)
            if spot_arm_ik_controller.deposit_pose is not None:
                AddMeshcatTriad(meshcat, "deposit_pose", length=0.1, radius=0.006, X_PT=spot_arm_ik_controller.deposit_pose)

            # Reset spot PID controller if requested (non-ideal solution)
            # if tidy_spot_planner.resetPID or spot_arm_ik_controller.resetPID:
            #     print("Resetting PID controller")
            #     inverse_dynamics_controller_context = inverse_dynamics_controller.GetMyContextFromRoot(context)
            #     inverse_dynamics_controller.set_integral_value(inverse_dynamics_controller_context, np.zeros((10, 1)))


            # inverse_dynamics_controller_context = inverse_dynamics_controller.GetMyContextFromRoot(context)
            # pid_output = inverse_dynamics_controller.get_output_port(0).Eval(inverse_dynamics_controller_context)
            # # print(f"Time: {context.get_time():.2f}")
            # # print("PID output (torques):", pid_output)
            # q9_pid_output_history.append(pid_output[9])
            # times.append(context.get_time())

        # simulator.set_monitor(PrintStates)
        # simulator.set_monitor(visualize_controller_poses_and_debug)

        meshcat.Flush()  # Wait for the large object meshes to get to meshcat.

        meshcat.StartRecording()
        simulator.AdvanceTo(15)
        print("Simulation has finished.")  # Print a message when the simulation is done
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

        # # Test segmentation on camera

        # object_detector.test_segmentation(object_detector_context, "frontleft")

        # # Test cropping from segmentation on frontleft camera
        # pcd_cropper_context = point_cloud_cropper.GetMyMutableContextFromRoot(context)
        # point_cloud_cropper.test_frontleft_crop_from_segmentation(pcd_cropper_context)

        # Test anygrasp on segmented point cloud
        # grasp_selector.test_anygrasp_frontleft_segmented_pcd(grasp_selector.GetMyMutableContextFromRoot(context))

        # Test grasp selection
        # grasp_pose = grasper.GetGraspSelection(grasper.GetMyMutableContextFromRoot(context))
        # AddMeshcatTriad(meshcat, "grasp_pose", length=0.1, radius=0.006, X_PT=grasp_pose)

        # # Keep meshcat alive
        meshcat.PublishRecording()

        # plt.figure()
        # plt.plot(times, q9_pid_output_history)
        # plt.xlabel('Time [s]')
        # plt.ylabel('q9_pid_output_history')
        # plt.title('q9 Force Outputs Over Time')
        # plt.grid(True)
        # plt.show()

        while not meshcat.GetButtonClicks("Stop meshcat"):
            pass

    finally:
        cleanup_resources()
