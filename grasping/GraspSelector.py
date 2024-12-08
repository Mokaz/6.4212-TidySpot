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

        # Output ports
        self.DeclareAbstractOutputPort("grasp_selection", lambda: AbstractValue.Make(RigidTransform()), self.SelectGrasp)

    def connect_ports(self, point_cloud_cropper, builder):
        builder.Connect(
            point_cloud_cropper.GetOutputPort("grasping_object_point_cloud"),
            self._point_cloud_input
        )

    def set_diagram(self, diagram):
        self.grasp_handler.set_diagram(diagram)

    def SelectGrasp(self, context: Context, output):
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
            print("ANYGRASP: No points in the specified limits.")
            return

        # if self.visualize:
        #     self.meshcat.SetObject("cropped_point_cloud", point_cloud, point_size=0.01)
        #self.visualize_pcd_with_grasps(points, colors)

        if self._use_anygrasp:
            g_best = self.grasp_handler.run_grasp(points, colors, lims=None, visualize=False)
            g_best = RigidTransform(RotationMatrix(g_best.rotation_matrix).multiply(RotationMatrix.MakeXRotation(-np.pi/2)), g_best.translation)
            output.set_value(g_best)
        else: # Antipodal
            g_best = self.grasp_handler.run_grasp(point_cloud, context, visualize=False)
            output.set_value(g_best)

        # visualize the best grasp
        # AddMeshcatTriad(self.meshcat, "best_grasp_pose", length=0.1, radius=0.005, X_PT=g_best)

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