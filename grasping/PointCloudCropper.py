import os
import logging
import numpy as np
from sklearn.cluster import DBSCAN

from pydrake.all import (
    AbstractValue,
    RigidTransform,
    Context,
    LeafSystem,
    PointCloud,
    DepthImageToPointCloud,
    DiagramBuilder,
    BaseField,
    Fields,
)
from typing import List, Tuple, Mapping

DO_DBSCAN_CLUSTERING = True

class PointCloudCropper(LeafSystem):
    def __init__(self, camera_names: List[str], meshcat=None):
        LeafSystem.__init__(self)
        self._camera_names = camera_names
        self.meshcat = meshcat

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
            self.CropPointCloudBySegmentation
        )

    def CropPointCloudBySegmentation(self, context: Context, output):
        segmentation_data = self._segmentation_data_input.Eval(context)
        segmentation_mask, camera_name = segmentation_data['segmentation_mask'], segmentation_data['camera_name']
        segmentation_mask = segmentation_mask.flatten()

        if segmentation_mask.size == 0:
            output.set_value(PointCloud(0))
            print("CropPointcloud: No segmentation mask found")
            return

        point_cloud = self.get_input_port(self._pcd_inputs_indexes[camera_name]).Eval(context)

        # DEBUG: Visualize the original point cloud
        # self.meshcat.SetObject("original_full_point_cloud", point_cloud)

        points = point_cloud.xyzs().T.astype(np.float32)
        colors = point_cloud.rgbs().T.astype(np.float32) / 255.0

        segmented_points = points[segmentation_mask]
        segmented_colors = colors[segmentation_mask]

        valid_mask = np.isfinite(segmented_points).all(axis=1)
        segmented_points = segmented_points[valid_mask]
        segmented_colors = segmented_colors[valid_mask]

        if DO_DBSCAN_CLUSTERING:
            # self.visualize_pcd(segmented_points, segmented_colors)
            # Assuming 'segmented_point_cloud' is a NumPy array of shape (N, 3)
            clustering = DBSCAN(eps=0.05, min_samples=10).fit(segmented_points)
            labels = clustering.labels_

            # Filter out noise points (labels == -1)
            non_noise = labels != -1
            filtered_points = segmented_points[non_noise]
            filtered_colors = segmented_colors[non_noise]

            # Select the largest cluster
            unique_labels, counts = np.unique(labels[non_noise], return_counts=True)
            largest_cluster = unique_labels[np.argmax(counts)]
            final_points = segmented_points[labels == largest_cluster]
            final_colors = segmented_colors[labels == largest_cluster]

            # self.visualize_pcd(final_points, final_colors)
        else:
            final_points = segmented_points
            final_colors = segmented_colors

        fields = Fields(BaseField.kXYZs | BaseField.kRGBs)
        segmented_point_cloud = PointCloud(final_points.shape[0], fields)
        segmented_point_cloud.mutable_xyzs()[:] = final_points.T
        segmented_point_cloud.mutable_rgbs()[:] = (final_colors.T * 255.0).astype(np.uint8)

        output.set_value(segmented_point_cloud)
        # print("PointCloudCropper: CropPointCloudBySegmentation, sending to PointCloudMapper")

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

    def test_frontleft_crop_from_segmentation(self, pcd_cropper_context: Context):
        segmentation_data = self._segmentation_data_input.Eval(pcd_cropper_context)
        segmentation_mask, camera_name = segmentation_data['segmentation_mask'], segmentation_data['camera_name']
        segmentation_mask = segmentation_mask.flatten()

        print(f"Segmentation mask shape: {segmentation_mask.shape}")

        if segmentation_mask.size == 0:
            print("CropPointcloud: No segmentation mask found")
            return

        point_cloud = self.get_input_port(self._pcd_inputs_indexes[camera_name]).Eval(pcd_cropper_context)

        # Get the points and colors
        points = point_cloud.xyzs().T.astype(np.float32)
        colors = point_cloud.rgbs().T.astype(np.float32) / 255.0

        valid_mask = np.isfinite(points).all(axis=1)

        points_viz = points[valid_mask]
        colors_viz = colors[valid_mask]

        self.visualize_pcd(points_viz, colors_viz)

        # Get the segmented points
        segmented_points = points[segmentation_mask]
        segmented_colors = colors[segmentation_mask]

        # Set the segmented point cloud
        fields = Fields(BaseField.kXYZs | BaseField.kRGBs)
        segmented_point_cloud = PointCloud(segmented_points.shape[0], fields)
        segmented_point_cloud.mutable_xyzs()[:] = segmented_points.T

        # Ensure colors are in the range [0, 255] and of type uint8
        segmented_point_cloud.mutable_rgbs()[:] = (segmented_colors.T * 255.0).astype(np.uint8)

        # Recover points and colors from Drake PointCloud
        segmented_points = segmented_point_cloud.xyzs().T.astype(np.float32)
        segmented_colors = segmented_point_cloud.rgbs().T.astype(np.float32) / 255.0

        valid_mask = np.isfinite(segmented_points).all(axis=1)

        segmented_points_viz = segmented_points[valid_mask]
        segmented_colors_viz = segmented_colors[valid_mask]

        self.visualize_pcd(segmented_points_viz, segmented_colors_viz)

    def visualize_pcd(self, points, colors=None):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])