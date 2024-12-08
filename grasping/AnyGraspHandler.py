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
            return None

        # Sort and initialize filtered grasp group
        gg = gg.nms().sort_by_score()

        if flip_before_calc:
            # Flip the z points back
            points[:, 2] *= -1
            gg.translations[:, 2] = -gg.translations[:, 2]
            # Reflect the rotation matrices about the xy plane
            S = np.array([
                [1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1]
            ])
            gg.rotation_matrices = gg.rotation_matrices * S

        return gg
