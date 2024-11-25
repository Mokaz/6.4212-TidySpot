from pydrake.all import Diagram
from typing import BinaryIO, Optional, Union, Tuple
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

# CUDA memory management
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