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
)
from grasping.grasp_utils import add_anygrasp_to_path
import open3d as o3d
from typing import List, Tuple, Mapping

class GraspSelector(LeafSystem):
    def __init__(self, use_anygrasp: bool, anygrasp_path: str = os.path.join(os.getcwd(), "anygrasp_sdk")):
        LeafSystem.__init__(self)
        self._use_anygrasp = use_anygrasp
        if use_anygrasp:
            self._setup_anygrasp(anygrasp_path)

        # Input ports
        self._point_cloud_input = self.DeclareAbstractInputPort("point_cloud_input", AbstractValue.Make(PointCloud(0)))

        # Output ports
        self.DeclareAbstractOutputPort("grasp_selection", lambda: AbstractValue.Make(RigidTransform()), self.SelectGrasp)

    def connect_ports(self, point_cloud_cropper, builder):
        builder.Connect(
            point_cloud_cropper.GetOutputPort("cropped_point_cloud"),
            self._point_cloud_input 
        )

    def SelectGrasp(self, context: Context, output):
        points = self._point_cloud_input.Eval(context).xyzs().T.astype(np.float32) 
        colors = self._point_cloud_input.Eval(context).rgbs().T.astype(np.float32) / 255.0
        
        if self._use_anygrasp:
            # gg_pick = self.run_anygrasp(points, colors, lims)
            output.set_value(RigidTransform())
            pass

    def _setup_anygrasp(self, anygrasp_path: str):
        try:
            add_anygrasp_to_path(anygrasp_path)

            from anygrasp_sdk.grasp_detection.gsnet import AnyGrasp

            cfgs = SimpleNamespace(
                checkpoint_path=os.path.join(anygrasp_path, 'grasp_detection', 'checkpoints', 'checkpoint_detection.tar'),
                max_gripper_width=1.0,
                gripper_height=0.03,
                top_down_grasp=False,  # Set to True if needed
                debug=False
            )

            self.anygrasp = AnyGrasp(cfgs)
            self.anygrasp.load_net()
        
        except ImportError as e:
            logging.error("AnyGrasp module not found. Ensure 'anygrasp_sdk' is correctly installed.")
            raise e
        except FileNotFoundError as e:
            logging.error(f"Checkpoint file not found: {e}")
            raise e
        except Exception as e:
            logging.error(f"Failed to initialize AnyGrasp: {e}")
            raise e

    def test_anygrasp_frontleft_pcd(self, to_point_cloud: Mapping[str, DepthImageToPointCloud], simulator_context: Context):
        frontleft_pcd = to_point_cloud["frontleft"].get_output_port().Eval(
            to_point_cloud["frontleft"].GetMyContextFromRoot(simulator_context)
        )

        points = frontleft_pcd.xyzs().T.astype(np.float32)         # Shape: (N, 3)
        colors = frontleft_pcd.rgbs().T.astype(np.float32) / 255.0 # Shape: (N, 3)

        valid_mask = np.isfinite(points).all(axis=1)

        points = points[valid_mask]
        colors = colors[valid_mask] 
        
        # self.visualize_pcd_with_grasps(points, colors)

        # Crops the point cloud for anygrasp, currently arbitrary values
        xmin, xmax = 0, 2  
        ymin, ymax = -1, 1  
        zmin, zmax = 0.2, 1.0   
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        mask = (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
            (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
        )

        points = points[mask]
        colors = colors[mask]

        self.run_anygrasp(points, colors, lims, visualize=True)

    def run_anygrasp(self, points, colors, lims, visualize=False):
        if not hasattr(self, 'anygrasp') or self.anygrasp is None:
            raise ValueError("AnyGrasp not initialized. Ensure 'use_anygrasp' is set to True.")

        gg, cloud = self.anygrasp.get_grasp(
            points,
            colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )

        if len(gg) == 0:
            print('No Grasp detected after collision detection!') # TODO: Implement retry mechanism
        else:
            gg = gg.nms().sort_by_score() # TODO: Grasp filtering based on orientation compared to Spot
            gg_pick = gg[0:20]
            print(gg_pick.scores)
            print('Top grasp score:', gg_pick[0].score)

        if visualize:
            self.visualize_pcd_with_grasps(points, colors, gg) # TODO: Send to meshcat?

        return gg_pick[0]

    def visualize_pcd_with_grasps(self, points, colors=None, gg=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries = [pcd]
        
        if gg is not None:
            grippers = gg.to_open3d_geometry_list()
            geometries.extend(grippers)
        
        o3d.visualization.draw_geometries(geometries)