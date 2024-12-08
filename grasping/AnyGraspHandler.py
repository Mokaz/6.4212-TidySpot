import os
import logging
from types import SimpleNamespace
from grasping.grasp_utils import add_anygrasp_to_path
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="WARNING:root:Failed to import ros dependencies in rigid_transforms.py")
warnings.filterwarnings("ignore", category=UserWarning, message="WARNING:root:autolab_core not installed as catkin package, RigidTransform ros methods will be unavailable")

class AnyGraspHandler:
    def __init__(self, anygrasp_path: str):
        self._setup_anygrasp(anygrasp_path)

    def _setup_anygrasp(self, anygrasp_path: str):
        try:
            add_anygrasp_to_path(anygrasp_path)
            import open3d as o3d
            from gsnet import AnyGrasp

            cfgs = SimpleNamespace(
                checkpoint_path=os.path.join(anygrasp_path, 'grasp_detection', 'checkpoints', 'checkpoint_detection.tar'),
                max_gripper_width=0.1,
                gripper_height=0.03,
                top_down_grasp=False,  # Set to True if needed
                debug=False
            )

            self.anygrasp = AnyGrasp(cfgs)
            self.anygrasp.load_net()
            print("AnyGrasp initialized successfully.")

        except ImportError as e:
            logging.error("AnyGrasp module not found. Ensure 'anygrasp_sdk' is correctly installed.")
            raise e
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found: {e}")
            raise e
        except Exception as e:
            logging.error(f"Failed to initialize AnyGrasp: {e}")
            raise e

    def set_diagram(self, diagram):
        self.diagram=diagram

    def run_grasp(self, points, colors, lims=None, flip_before_calc=True, visualize=False):
        """
        Runs the AnyGrasp grasp detection on the provided point cloud.

        Args:
            points (np.ndarray): Array of point coordinates.
            colors (np.ndarray): Array of point colors.
            lims (list): List of limits [xmin, xmax, ymin, ymax, zmin, zmax].
            visualize (bool, optional): Flag to visualize the results. Defaults to False.

        Returns:
            Grasp: The selected grasp.
        """

        from graspnetAPI import GraspGroup

        if self.anygrasp is None:
            raise ValueError("AnyGrasp not initialized.")

        # Flip the z points in the points array
        if flip_before_calc:
            points[:, 2] *= -1
            if lims is not None:
                lims[4], lims[5] = -lims[5], -lims[4]

        kwargs = {
            'points': points,
            'colors': colors,
            'apply_object_mask': True,
            'dense_grasp': False,
            'collision_detection': True
        }
        if lims is not None:
            kwargs['lims'] = lims

        print('Running AnyGrasp...')
        gg, cloud = self.anygrasp.get_grasp(**kwargs)

        if gg is None or len(gg) == 0:
            print('No Grasp detected after collision detection!')

        # Sort and initialize filtered grasp group
        gg = gg.nms().sort_by_score()
        filter_gg = GraspGroup()

        for g in gg:
            R = g.rotation_matrix
            gripper_x_axis = R[:, 0]
            gripper_y_axis = R[:, 1]

            # Compute rotation matrix to align gripper x-axis with the positive z-direction
            target_axis = np.array([0.0, 0.0, 1.0])  # Desired x-axis direction
            current_axis = gripper_x_axis
            cross_prod = np.cross(current_axis, target_axis)
            dot_prod = np.dot(current_axis, target_axis)
            
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

            # Check alignment with thresholds
            x_alignment = np.dot(gripper_x_axis, target_axis)
            y_alignment = np.dot(gripper_y_axis, np.array([1.0, 0.0, 0.0]))

            if x_alignment > 0.9 and y_alignment > 0.9:  # Adjust thresholds if needed
                filter_gg.add(g)

        if len(filter_gg) == 0:
            print('No Grasp detected after filtering!')

        # Select top grasps based on score
        if len(filter_gg) < 10:
            gg_pick = filter_gg.sort_by_score()[0:len(filter_gg)]
        else:
            gg_pick = filter_gg.sort_by_score()[0:10]

        print('Top grasp calculated with score:', gg_pick[0].score)

        if flip_before_calc:
            # Flip the z points back
            points[:, 2] *= -1
            gg_pick.translations[:, 2] = -gg_pick.translations[:, 2]
            # Reflect the rotation matrices about the xy plane
            S = np.array([
                [1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1]
            ])
            gg_pick.rotation_matrices = gg_pick.rotation_matrices * S

        if visualize:
            self.visualize_pcd_with_grasps(points, colors, gg_pick)
            self.visualize_pcd_with_single_grasp(points, gg_pick[0], colors)

        return gg_pick[0]

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

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geometries.append(axis)

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
