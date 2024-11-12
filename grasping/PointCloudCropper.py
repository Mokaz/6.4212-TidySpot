import os
import logging
import numpy as np
from types import SimpleNamespace
from pydrake.all import (
    AbstractValue,
    RigidTransform,
    Context,
    LeafSystem,
    PointCloud,
    DepthImageToPointCloud,
    DiagramBuilder
)
import open3d as o3d
from typing import List, Tuple, Mapping

class PointCloudCropper(LeafSystem):
    def __init__(self, camera_names: List[str]):
        LeafSystem.__init__(self)
        self._camera_names = camera_names

        # Input ports
        self._pcd_inputs_indexes = {
            camera_name: self.DeclareAbstractInputPort(
                f"{camera_name}.point_cloud",
                AbstractValue.Make(PointCloud(0))
            ).get_index()
            for camera_name in camera_names
        }

        self._segmentation_data_input = self.DeclareAbstractInputPort(
                "segmentation_data",
                AbstractValue.Make({
                    "segmentation_mask": np.array([]),
                    "camera_name": ""
                })
        )

        # Output ports
        self.DeclareAbstractOutputPort(
            "cropped_point_cloud",
            lambda: AbstractValue.Make(PointCloud(0)),
            self.CropPointcloud
        )

    def CropPointcloud(self, context: Context, output):
        segmentation_data = self._segmentation_data_input.Eval(context)
        segmentation_mask, camera_name = segmentation_data['segmentation_mask'], segmentation_data['camera_name']

        if segmentation_data[segmentation_mask].size == 0:
            output.set_value(PointCloud(0))
            print("CropPointcloud: No segmentation mask found")
            return
        
        point_cloud = self.EvalAbstractInput(context, self._pcd_inputs_indexes[segmentation_data[camera_name]])

        # Get the points and colors
        points = point_cloud.xyzs().T.astype(np.float32)
        colors = point_cloud.rgbs().T.astype(np.float32) / 255.0

        # Get the segmented points
        segmented_points = points[segmentation_mask]
        segmented_colors = colors[segmentation_mask]

        # Set the segmented point cloud
        segmented_point_cloud = PointCloud(0)
        segmented_point_cloud.set_xyzs(segmented_points.T)
        segmented_point_cloud.set_rgbs(segmented_colors.T * 255.0)

        output.set_value(segmented_point_cloud)

    def connect_ports(self, to_point_cloud: Mapping[str, DepthImageToPointCloud], object_detector, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            builder.Connect(
                to_point_cloud[camera_name].GetOutputPort(f"point_cloud"), 
                self.get_input_port(self._pcd_inputs_indexes[camera_name])
            )

        builder.Connect(
            object_detector.GetOutputPort("segmentation_data"),
            self._segmentation_data_input
        )