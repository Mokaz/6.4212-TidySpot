import os
import logging
import numpy as np
from sklearn.cluster import DBSCAN

from pydrake.all import (
    AbstractValue,
    RigidTransform,
    Context,
    Concatenate,
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

        self._object_detection_segmentations_input = self.DeclareAbstractInputPort(
                "object_detection_segmentations",
                AbstractValue.Make({
                    "camera_name": np.zeros((0, 0), dtype=bool)
                })
        )

        self._grasping_object_segmentation_input = self.DeclareAbstractInputPort(
                "grasping_object_segmentation",
                AbstractValue.Make({
                    "camera_name": np.zeros((0, 0), dtype=bool)
                })
        )

        # Output ports
        self.DeclareAbstractOutputPort(
            "object_detection_cropped_point_clouds",
            lambda: AbstractValue.Make(PointCloud(0)),
            self.CropObjectDetectionPointClouds
        )

        self.DeclareAbstractOutputPort(
            "grasping_object_point_cloud",
            lambda: AbstractValue.Make(PointCloud(0)),
            self.CropGraspingObjectPointCloud
        )

    def CropObjectDetectionPointClouds(self, context: Context, output):
        segmentation_masks_dict = self._object_detection_segmentations_input.Eval(context)
        cropped_point_cloud = self.CropPointCloudBySegmentation(context, output, segmentation_masks_dict)
        output.set_value(cropped_point_cloud)

    def CropGraspingObjectPointCloud(self, context: Context, output):
        segmentation_masks_dict = self._grasping_object_segmentation_input.Eval(context)
        cropped_object_pcd = self.CropPointCloudBySegmentation(context, output, segmentation_masks_dict)
        output.set_value(cropped_object_pcd)

    def CropPointCloudBySegmentation(self, context: Context, output, segmentation_masks_dict):
        if not segmentation_masks_dict:
            output.set_value(PointCloud(0))
            print("CropPointcloud: No segmentation masks received.")
            return
        
        cropped_point_cloud_list = []
        
        for camera_name, mask in segmentation_masks_dict.items():
            segmentation_mask = mask.flatten()

            point_cloud = self.get_input_port(self._pcd_inputs_indexes[camera_name]).Eval(context)

            # DEBUG: Visualize the original point cloud
            # self.meshcat.SetObject("original_full_point_cloud", point_cloud)

            # print(f"CropPointcloud: Segmentation mask shape: {segmentation_mask.shape}")
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

            cropped_point_cloud_list.append(segmented_point_cloud)

        cropped_point_cloud = Concatenate(cropped_point_cloud_list)

        print("PointCloudCropper: CropPointCloudBySegmentation, sending to PointCloudMapper")
        return cropped_point_cloud

    def connect_ports(self, to_point_cloud: Mapping[str, DepthImageToPointCloud], object_detector, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            builder.Connect(
                to_point_cloud[camera_name].GetOutputPort(f"point_cloud"),
                self.get_input_port(self._pcd_inputs_indexes[camera_name])
            )

        builder.Connect(
            object_detector.GetOutputPort("object_detection_segmentations"),
            self._object_detection_segmentations_input
        )

        builder.Connect(
            object_detector.GetOutputPort("grasping_object_segmentation"),
            self._grasping_object_segmentation_input
        )

    def visualize_pcd(self, points, colors=None):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])