import os
import logging
from types import SimpleNamespace
from grasping.grasp_utils import add_anygrasp_to_path
import numpy as np
import open3d as o3d

class AnyGraspHandler:
    def __init__(self, anygrasp_path: str):
        self._setup_anygrasp(anygrasp_path)

    def _setup_anygrasp(self, anygrasp_path: str):
        try:
            add_anygrasp_to_path(anygrasp_path)
            from anygrasp_sdk.grasp_detection.gsnet import AnyGrasp

            cfgs = SimpleNamespace(
                checkpoint_path=os.path.join(anygrasp_path, 'grasp_detection', 'checkpoints', 'checkpoint_detection.tar'),
                max_gripper_width=0.8,
                gripper_height=0.3,
                top_down_grasp=True,  # Set to True if needed
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

    def run_anygrasp(self, points, colors, lims=None, flip_before_calc=True, visualize=False):
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

        if len(gg) == 0:
            print('No Grasp detected after collision detection!') # TODO: Implement retry mechanism
        else:
            gg = gg.nms().sort_by_score() # TODO: Grasp filtering based on orientation compared to Spot
            gg_pick = gg[0:10]
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
            self.visualize_pcd_with_grasps(points, colors, gg_pick) # TODO: Send to meshcat?
            self.visualize_pcd_with_single_grasp(points, gg_pick[0], colors)

        return gg_pick

    def visualize_pcd_with_grasps(self, points, colors=None, gg=None):
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries = [pcd, g.to_open3d_geometry()]
        
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geometries.append(axis)
        
        o3d.visualization.draw_geometries(geometries)
