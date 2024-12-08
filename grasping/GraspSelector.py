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
    RotationMatrix,
)

from manipulation.meshcat_utils import AddMeshcatTriad
from grasping.grasp_utils import add_anygrasp_to_path
from typing import List, Tuple, Mapping

VISUALIZE_GRASPS = False

class GraspSelector(LeafSystem):
    def __init__(self, use_anygrasp: bool, plant, scene_graph, meshcat, anygrasp_path: str = os.path.join(os.getcwd(), "third_party/anygrasp_sdk")):
        LeafSystem.__init__(self)
        self._use_anygrasp = use_anygrasp
        self.grasp_handler = None
        self.meshcat = meshcat
        self.visualize = True

        if use_anygrasp:
            import open3d as o3d
            from grasping.AnyGraspHandler import AnyGraspHandler
            self.grasp_handler = AnyGraspHandler(anygrasp_path)
        else:
            # If not using anygrasp, then we are using antipodal
            from grasping.AntipodalGraspHandler import AntipodalGraspHandler
            self.grasp_handler = AntipodalGraspHandler(plant, scene_graph, meshcat) # TODO, change anygrasp_handler to just grasp handler to abstract it


        # Input ports
        self._point_cloud_input = self.DeclareAbstractInputPort("point_cloud_input", AbstractValue.Make(PointCloud(0)))
        self.spot_state_estimated_input = self.DeclareVectorInputPort("spot.state_estimated",20)
        self.current_object_location_input = self.DeclareVectorInputPort("current_object_location", 2)

        # Output ports
        self.DeclareAbstractOutputPort("grasp_selection", lambda: AbstractValue.Make(RigidTransform()), self.SelectGrasp)

    def connect_ports(self, station, point_cloud_cropper, builder):
        builder.Connect(
            point_cloud_cropper.GetOutputPort("grasping_object_point_cloud"),
            self._point_cloud_input
        )
        builder.Connect(
            station.GetOutputPort("spot.state_estimated"),
            self.GetInputPort("spot.state_estimated"),
        )

    def set_diagram(self, diagram):
        self.grasp_handler.set_diagram(diagram)

    def SelectGrasp(self, context: Context, output):
        curr_spot_xy = self.spot_state_estimated_input.Eval(context)[:2]
        current_object_location = self.current_object_location_input.Eval(context)

        point_cloud = self._point_cloud_input.Eval(context)
        points = point_cloud.xyzs().T.astype(np.float32)
        valid_mask = np.isfinite(points).all(axis=1)

        if point_cloud.has_rgbs():
            colors = point_cloud.rgbs().T.astype(np.float32) / 255.0
            colors = colors[valid_mask]
        else:
            colors = None

        points = points[valid_mask]

        if points.shape[0] == 0:
            print("SelectGrasp: No points in the specified limits.")
            return

        # if self.visualize:
        #     self.meshcat.SetObject("cropped_point_cloud", point_cloud, point_size=0.01)
        #self.visualize_pcd_with_grasps(points, colors)

        if self._use_anygrasp:
            gg = self.grasp_handler.run_grasp(points, colors, lims=None, visualize=False)

            if gg is None:
                print('ANYGRASP: No grasps found!')
                output.set_value(RigidTransform())
                return

            # Filter grasps based on the spot location and object location
            gg = self.filter_best_topdown_grasps(gg, curr_spot_xy, current_object_location, vertical_alignment_threshold=0.9, spot_to_object_vector_alignment_threshold=0.9, filter_based_on_spot_to_object_vector=True)
            if gg is None:
                print('ANYGRASP: No grasps found after filtering!')
                output.set_value(RigidTransform())
                return

            if VISUALIZE_GRASPS:
                self.visualize_pcd_with_grasps(points, colors, gg)
                self.visualize_pcd_with_single_grasp(points, gg[0], colors)

            g_best = RigidTransform(RotationMatrix(gg[0].rotation_matrix).multiply(RotationMatrix.MakeXRotation(-np.pi/2)), gg[0].translation)
            output.set_value(g_best)
        else: # Antipodal
            g_best = self.grasp_handler.run_grasp(point_cloud, context, visualize=False)
            output.set_value(g_best)

        # visualize the best grasp
        AddMeshcatTriad(self.meshcat, "best_grasp_pose", length=0.1, radius=0.005, X_PT=g_best)

    def filter_best_topdown_grasps(self, gg, spot_location, object_location, vertical_alignment_threshold, spot_to_object_vector_alignment_threshold, filter_based_on_spot_to_object_vector=True):
        from graspnetAPI import GraspGroup
        filter_gg = GraspGroup()

        for g in gg:
            R = g.rotation_matrix
            gripper_x_axis = R[:, 0]
            gripper_y_axis = R[:, 1]

            # Compute rotation matrix to align gripper x-axis with the negative z-direction
            target_axis = np.array([0.0, 0.0, -1.0])  # Desired x-axis direction
            current_axis = gripper_x_axis
            cross_prod = np.cross(current_axis, target_axis)
            dot_prod = np.dot(current_axis, target_axis)

            # Aligns x-axis with the negative z-axis
            if np.linalg.norm(cross_prod) > 1e-6:  # Avoid singularities
                axis = cross_prod / np.linalg.norm(cross_prod)  # Normalize rotation axis
                angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))  # Angle between the vectors

                # Construct rotation matrix using Rodrigues' rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                g.rotation_matrix = R_align @ R
            else:
                # If already aligned or nearly aligned, no rotation needed
                g.rotation_matrix = R

            # Recalculate axes after rotation
            R = g.rotation_matrix
            gripper_x_axis = R[:, 0]
            gripper_y_axis = R[:, 1]

            # Compute spot to object vector 
            spot_to_object_vector = np.append((object_location - spot_location) / np.linalg.norm(object_location - spot_location), [0.0])

            # Check alignment with thresholds
            vertical_alignment = np.dot(gripper_x_axis, target_axis)
            spot_to_object_alignment = np.dot(gripper_y_axis, spot_to_object_vector)

            if filter_based_on_spot_to_object_vector:
                if vertical_alignment > vertical_alignment_threshold and spot_to_object_alignment > spot_to_object_vector_alignment_threshold: 
                    filter_gg.add(g)
            else:
                if vertical_alignment > vertical_alignment_threshold:
                    filter_gg.add(g)

        if len(filter_gg) == 0:
            print('No Grasp detected after filtering!')
            return None

        return filter_gg
            

    def visualize_pcd_with_grasps(self, points, colors=None, gg=None):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        geometries = [pcd]

        if gg is not None:
            grippers = gg.to_open3d_geometry_list()
            geometries.extend(grippers)

        # Add coordinate axis for better visualization
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geometries.extend([axis])
        o3d.visualization.draw_geometries(geometries)

    def visualize_pcd_with_single_grasp(self, points: np.ndarray, g, colors = None):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        geometries = [pcd, g.to_open3d_geometry()]

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(axis)

        o3d.visualization.draw_geometries(geometries)