from pydrake.all import Diagram
from typing import BinaryIO, Optional, Union, Tuple, List
import random
import pydot
import logging
import torch
import sys
import signal

import os
import sys
import typing

from pydrake.all import (
    ApplyCameraConfig,
    ApplyLcmBusConfig,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    BaseField,
    CameraInfo,
    DepthImageToPointCloud,
    Diagram,
    DiagramBuilder,
    GetScopedFrameByName,
    Meshcat,
    MeshcatPointCloudVisualizer,
    OutputPort,
    Rgba,
    Sphere,
    RigidTransform,
)

from manipulation.systems import ExtractPose
from manipulation.station import Scenario

### Filter out Drake warnings ###
class DrakeWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "has its own materials, but material properties have been defined as well" in msg or \
           "material [ 'wire_088144225' ] not found in .mtl" in msg:
            return False
        return True


def export_diagram_as_svg(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)

### CUDA memory management ###
def cleanup_resources():
    print('Cleaning up resources...')
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Force garbage collection
    import gc
    gc.collect()

def signal_handler(sig, frame):
    print('Termination signal received.')
    cleanup_resources()
    sys.exit(0)

def register_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

### Meshcat ###
def add_sphere_to_meshcat_xy_plane(meshcat, name, position, radius=0.05, rgba=[1, 0, 0, 1]):
    meshcat.SetObject(
        name,
        Sphere(radius=radius),
        Rgba(*rgba)
    )
    meshcat.SetTransform(
        name,
        RigidTransform([position[0], position[1], 0])
    )

### Drake ###
import numpy as np

def get_bin_translation(scenario, bin_link_name):
    """
    Extracts the translation of planar_bin::bin_base from the scenario directives.

    Parameters:
        scenario (object): The scenario object containing directives.

    Returns:
        numpy.ndarray: Translation array of the bin if found, otherwise [0, 0, 0].
    """
    bin_weld_directive = next(
        (directive for directive in scenario.directives if
         directive.add_weld and directive.add_weld.child == bin_link_name),
        None
    )

    if bin_weld_directive:
        return bin_weld_directive.add_weld.X_PC.translation
    else:
        print("Weld directive for planar_bin::bin_base not found. Setting to default bin location: [0, 0, 0]")
        return np.array([0, 0, 0])

def AddPointClouds(
    *,
    scenario: Scenario,
    station: Diagram,
    builder: DiagramBuilder,
    poses_output_port: OutputPort = None,
    meshcat: Meshcat = None,
) -> typing.Mapping[str, DepthImageToPointCloud]:
    """
    Adds one DepthImageToPointCloud system to the `builder` for each camera in `scenario`, and connects it to the respective camera station output ports.

    Modified for the TidySpot project to include X_BD transformation of depth camera.

    Args:
        scenario: A Scenario structure, populated using the `LoadScenario` method.

        station: A HardwareStation system (e.g. from MakeHardwareStation) that has already been added to `builder`.

        builder: The DiagramBuilder containing `station` into which the new systems will be added.

        poses_output_port: (optional) HardwareStation will have a body_poses output port iff it was created with `hardware=False`. Alternatively, one could create a MultibodyPositionsToGeometryPoses system to consume the position measurements; this optional input can be used to support that workflow.

        meshcat: If not None, then a MeshcatPointCloudVisualizer will be added to the builder using this meshcat instance.

    Returns:
        A mapping from camera name to the DepthImageToPointCloud system.
    """
    to_point_cloud = dict()
    for _, config in scenario.cameras.items():
        if not config.depth:
            return

        plant = station.GetSubsystemByName("plant")
        # frame names in local variables:
        # P for parent frame, B for base frame, C for camera frame.

        # Extract the camera extrinsics from the config struct.
        P = (
            GetScopedFrameByName(plant, config.X_PB.base_frame)
            if config.X_PB.base_frame
            else plant.world_frame()
        )
        X_PC = config.X_PB.GetDeterministicValue() @ config.X_BD.GetDeterministicValue()

        # convert mbp frame to geometry frame
        body = P.body()
        plant.GetBodyFrameIdIfExists(body.index())
        # assert body_frame_id.has_value()

        X_BP = P.GetFixedPoseInBodyFrame()
        X_BC = X_BP @ X_PC

        intrinsics = CameraInfo(
            config.width,
            config.height,
            config.focal_x(),
            config.focal_y(),
            config.principal_point()[0],
            config.principal_point()[1],
        )

        to_point_cloud[config.name] = builder.AddSystem(
            DepthImageToPointCloud(
                camera_info=intrinsics,
                fields=BaseField.kXYZs | BaseField.kRGBs,
            )
        )
        to_point_cloud[config.name].set_name(f"{config.name}.point_cloud")

        builder.Connect(
            station.GetOutputPort(f"{config.name}.depth_image"),
            to_point_cloud[config.name].depth_image_input_port(),
        )
        builder.Connect(
            station.GetOutputPort(f"{config.name}.rgb_image"),
            to_point_cloud[config.name].color_image_input_port(),
        )

        if poses_output_port is None:
            # Note: this is a cheat port; it will only work in single process
            # mode.
            poses_output_port = station.GetOutputPort("body_poses")

        camera_pose = builder.AddSystem(ExtractPose(int(body.index()), X_BC))
        camera_pose.set_name(f"{config.name}.pose")
        builder.Connect(
            poses_output_port,
            camera_pose.get_input_port(),
        )
        builder.Connect(
            camera_pose.get_output_port(),
            to_point_cloud[config.name].GetInputPort("camera_pose"),
        )

        if meshcat:
            # Send the point cloud to meshcat for visualization, too.
            point_cloud_visualizer = builder.AddSystem(
                MeshcatPointCloudVisualizer(meshcat, f"{config.name}.cloud")
            )
            builder.Connect(
                to_point_cloud[config.name].point_cloud_output_port(),
                point_cloud_visualizer.cloud_input_port(),
            )

    return to_point_cloud



def convert_to_grid_coordinates(x: float, y: float, resolution: float, shape: Tuple) -> Tuple[int, int]:
    """
    Converts a world coordinate to grid coordinates.

    x, y: World coordinates
    Returns:
        ix, iy: Grid coordinates
    """
    ix = int(round(x / resolution)) + (shape[0] // 2)
    iy = int(round(y / resolution)) + (shape[1] // 2)
    return ix, iy

def convert_to_world_coordinates(ix: int, iy: int, resolution: float, shape: Tuple) -> Tuple[float, float]:
    """
    Converts grid coordinates to world coordinates.

    ix, iy: Grid coordinates
    Returns:
        x, y: World coordinates
    """
    x = (ix - (shape[0] // 2)) * resolution
    y = (iy - (shape[1] // 2)) * resolution
    return x, y

def clutter_gen(num_items: int, spawn_area: Tuple[float, float], goal_file: str, forbidden_areas: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []):
    ycb = ["003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf", 
           "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"]
    ycb_bases = ["base_link_cracker", "base_link_sugar", "base_link_soup", 
                 "base_link_mustard", "base_link_gelatin", "base_link_meat"]
    
    # Step 0: Check if we are in the right directory
    current_directory = os.getcwd()
    if not current_directory.endswith("6.4212_TidySpot"):
        goal_path = os.path.join("6.4212_TidySpot", goal_file)
    else: 
        goal_path = goal_file

    # Step 1: Generate empty yaml
    yaml_content = "directives:"

    # Step 2: Generate random positions, avoiding forbidden rectangular areas
    rng = random.Random()  # Initialize a random generator
    for i in range(num_items):
        object_num = rng.randint(0, len(ycb) - 1)
        
        # Generate a position and check if it's within any forbidden rectangular area
        in_forbidden_area = True
        while in_forbidden_area:
            in_forbidden_area = False
            x_pos = rng.uniform(-spawn_area[0], spawn_area[0])
            y_pos = rng.uniform(-spawn_area[1], spawn_area[1])
            
            # Check if the position is in any forbidden rectangular area
            for (corner1, corner2) in forbidden_areas:
                x_min = min(corner1[0], corner2[0])
                x_max = max(corner1[0], corner2[0])
                y_min = min(corner1[1], corner2[1])
                y_max = max(corner1[1], corner2[1])
                
                # Check if (x_pos, y_pos) is within the rectangle
                if x_min <= x_pos <= x_max and y_min <= y_pos <= y_max:
                    in_forbidden_area = True
                    break
            
            # If not in a forbidden area, proceed
            if not in_forbidden_area:
                break

        yaml_content += f"""
- add_model:
    name: obj_{i}
    file: package://manipulation/hydro/{ycb[object_num]}
    default_free_body_pose:
        {ycb_bases[object_num]}:
            translation: [{x_pos}, {y_pos}, 2]"""
        if object_num == 3:
            yaml_content += """
            rotation: !Rpy { deg: [0, 0, 0]}"""
        else:
            yaml_content += """
            rotation: !Rpy { deg: [90, 180, 0]}"""

    # Step 3: Write the modified content to the goal file
    with open(goal_path, 'w') as file:
        file.write(yaml_content)
    print(f"File saved to {goal_path}")

    return goal_file
