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
    RotationMatrix
)
from grasping.grasp_utils import add_anygrasp_to_path
from grasping.AnyGraspHandler import AnyGraspHandler
import open3d as o3d
from typing import List, Tuple, Mapping

class GraspSelector(LeafSystem):
    def __init__(self, use_anygrasp: bool, anygrasp_path: str = os.path.join(os.getcwd(), "third_party/anygrasp_sdk")):
        LeafSystem.__init__(self)
        self._use_anygrasp = use_anygrasp
        self.anygrasp_handler = None

        if use_anygrasp:
            self.anygrasp_handler = AnyGraspHandler(anygrasp_path)

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
        point_cloud = self._point_cloud_input.Eval(context)
        points = point_cloud.xyzs().T.astype(np.float32) 
        colors = point_cloud.rgbs().T.astype(np.float32) / 255.0

        valid_mask = np.isfinite(points).all(axis=1)

        points = points[valid_mask]
        colors = colors[valid_mask] 

        if points.shape[0] == 0:
            print("ANYGRASP: No points in the specified limits.")
            return
        
        # self.visualize_pcd_with_grasps(points, colors)
        
        if self._use_anygrasp:
            gg_ten_best = self.anygrasp_handler.run_anygrasp(points, colors, lims=None, visualize=False) # TODO: Implement more filtering
            g_best = gg_ten_best[0]
            output.set_value(RigidTransform(RotationMatrix(g_best.rotation_matrix), g_best.translation))

    def test_anygrasp_frontleft_segmented_pcd(self, grasp_selector_context: Context):
        frontleft_pcd = self._point_cloud_input.Eval(grasp_selector_context)
        points = frontleft_pcd.xyzs().T.astype(np.float32)         # Shape: (N, 3)
        colors = frontleft_pcd.rgbs().T.astype(np.float32) / 255.0

        valid_mask = np.isfinite(points).all(axis=1)

        points = points[valid_mask]
        colors = colors[valid_mask] 

        if points.shape[0] == 0:
            print("ANYGRASP: No points in the specified limits.")
            return

        # self.visualize_pcd_with_grasps(points, colors)

        # Limits for the point cloud (optional)

        xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
        ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
        zmin, zmax = np.min(points[:, 2]), np.max(points[:, 2])

        z_mid = (zmax - zmin) / 2

        zmin, zmax = zmin + z_mid, zmax + z_mid

        def create_xy_plane(z, size, resolution):
            # Generate grid points
            x = np.linspace(-size, size, resolution)
            y = np.linspace(-size, size, resolution)
            xx, yy = np.meshgrid(x, y)
            zz = np.full_like(xx, z)
            
            # Flatten the arrays and combine into vertex coordinates
            vertices = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
            
            # Create triangles
            triangles = []
            for i in range(resolution - 1):
                for j in range(resolution - 1):
                    # Calculate the index of the four corners of the current grid cell
                    idx = i * resolution + j
                    idx_right = idx + 1
                    idx_down = idx + resolution
                    idx_down_right = idx_down + 1
                    
                    # Define two triangles for each grid cell
                    triangles.append([idx, idx_down, idx_right])
                    triangles.append([idx_right, idx_down, idx_down_right])
            
            triangles = np.array(triangles)
            
            # Create the mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            
            # Optionally compute normals for better visualization
            mesh.compute_vertex_normals()
            
            return mesh

        zmin_plane = create_xy_plane(zmin, 1, 10)
        zmax_plane = create_xy_plane(zmax, 1, 10)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        geometries = [pcd, zmin_plane, zmax_plane]

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geometries.extend([axis])
        o3d.visualization.draw_geometries(geometries)

        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        self.anygrasp_handler.run_anygrasp(points, colors, lims, flip_before_calc=True, visualize=True)

    def test_anygrasp_frontleft_pcd(self, to_point_cloud: Mapping[str, DepthImageToPointCloud], simulator_context: Context):
        frontleft_pcd = to_point_cloud["frontleft"].get_output_port().Eval(
            to_point_cloud["frontleft"].GetMyContextFromRoot(simulator_context)
        )
        # X_WC = to_point_cloud["frontleft"].camera_pose_input_port().Eval(
        #     to_point_cloud["frontleft"].GetMyContextFromRoot(simulator_context)
        # )

        # R_WC = X_WC.rotation()

        # R_WC_new = R_WC.multiply(RotationMatrix.MakeYRotation(-np.pi / 2))

        # X_WC.set_rotation(R_WC_new)

        points = frontleft_pcd.xyzs().T.astype(np.float32)         # Shape: (N, 3)
        colors = frontleft_pcd.rgbs().T.astype(np.float32) / 255.0 # Shape: (N, 3)

        valid_mask = np.isfinite(points).all(axis=1)

        points = points[valid_mask]
        colors = colors[valid_mask] 
        
        # self.visualize_pcd_with_grasps(points, colors)

        # Crops the point cloud for anygrasp, crackers in front
        xmin, xmax = 0, 1  
        ymin, ymax = -1, 1  
        zmin, zmax = 0, 1
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        mask = (
            (points[:, 0] >= xmin) & (points[:, 0] <= xmax) &
            (points[:, 1] >= ymin) & (points[:, 1] <= ymax) &
            (points[:, 2] >= zmin) & (points[:, 2] <= zmax)
        )

        points = points[mask]
        colors = colors[mask]

        if points.shape[0] == 0:
            print("ANYGRASP: No points in the specified limits.")
            return
        
        # self.visualize_pcd_with_grasps(points, colors)

        # points = X_WC.inverse().multiply(points.T).T.astype(np.float32)

        # self.visualize_pcd_with_grasps(points, colors)

        self.anygrasp_handler.run_anygrasp(points, colors, visualize=True)

    def visualize_pcd_with_grasps(self, points, colors=None, gg=None):
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