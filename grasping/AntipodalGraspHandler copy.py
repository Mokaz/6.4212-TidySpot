import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    Evaluate,
    Jacobian,
    MathematicalProgram,
    Solve,
    Variable,
    atan,
    cos,
    eq,
    sin,
    Box,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    StartMeshcat,
)
from scipy.spatial import KDTree

from manipulation import running_as_notebook
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

class AntipodalGraspHandler:
    def __init__(self, plant, scene_graph, meshcat):
        self.meshcat = meshcat
        self.plant = plant
        self.scene_graph = scene_graph
        self.meshcat.Delete()  # Clears previous visualizations

    def compute_darboux_frame(self, index, sample_indices, pcd, kdtree, ball_radius=0.002, max_nn=50):
        points = pcd.xyzs()[:,sample_indices]
        pcd.EstimateNormals(radius=0.01, num_closest=30)
        normals = pcd.normals()[:,sample_indices]

        query = points[:, index]
        (distances, indices) = kdtree.query(query, k=max_nn, distance_upper_bound=ball_radius)
        valid_indices = indices[np.where(distances < np.inf)]

        neighbor_normals = normals[:, valid_indices]
        N = neighbor_normals @ neighbor_normals.T
        eigenvalues, eigenvectors = np.linalg.eig(N)
        order = np.argsort(eigenvalues)
        V = eigenvectors[:, order[::-1]]
        v_1 = V[:, 0]
        v_2 = V[:, 1]
        v_3 = V[:, 2]

        if v_1.dot(normals[:, index]) > 0:
            v_1 *= -1

        R = np.column_stack((v_2, v_1, v_3))
        R = R @ np.diag([1, 1, np.linalg.det(R)])

        X_WF = RigidTransform(RotationMatrix(R), query)
        return X_WF

    def draw_grasp_candidate(self, X_G, prefix="gripper", draw_frames=True, refresh=False):
        if refresh:
            self.meshcat.Delete()
        builder = DiagramBuilder()
        plant, scene_graph = self.plant, self.scene_graph
        parser = Parser(plant)
        ConfigureParser(parser)
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
        if draw_frames:
            AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
        plant.Finalize()

        params = MeshcatVisualizerParams()
        params.prefix = prefix
        meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat, params)
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(context)

    def find_minimum_distance(self, sampled_points, X_WG, context):
        y_grid = np.linspace(-0.05, 0.05, 10)

        max_distance = 0
        X_WGnew = None

        for delta in y_grid:
            X_WGsearch = X_WG.multiply(RigidTransform([0, delta, 0]))
            distance = self.compute_sdf(sampled_points, context, X_WGsearch)
            if distance >= 0:
                max_distance = distance
                X_WGnew = X_WGsearch
            elif distance < 0:
                break

        if X_WGnew is None:
            return np.nan, None
        return max_distance, X_WGnew

    def check_nonempty(self, sampled_points, X_WG, visualize=False):
        pcd_W_np = sampled_points
        crop_min = np.array([-0.05, 0.1, -0.00625])
        crop_max = np.array([0.05, 0.1125, 0.00625])

        # Transform the point cloud to the gripper frame
        X_GW = X_WG.inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)

        # Perform vectorized comparison to check if points lie within bounds
        within_bounds = np.logical_and(
            np.all(pcd_G_np >= crop_min[:, None], axis=0),
            np.all(pcd_G_np <= crop_max[:, None], axis=0)
        )

        # Check if there are any points within the bounds
        is_nonempty = np.any(within_bounds)

        # if visualize:
        #     self.visualize_nonempty(pcd, pcd_G_np, crop_min, crop_max)
        return is_nonempty

    def visualize_nonempty(self, pcd, pcd_G_np, crop_min, crop_max):
        # Visualize the results in Meshcat
        self.meshcat.Delete()
        pcd_G = PointCloud(pcd.size())
        pcd_G.mutable_xyzs()[:] = pcd_G_np

        self.draw_grasp_candidate(RigidTransform())
        self.meshcat.SetObject("cloud", pcd_G)

        box_length = crop_max - crop_min
        box_center = (crop_max + crop_min) / 2.0
        self.meshcat.SetObject(
            "closing_region",
            Box(box_length[0], box_length[1], box_length[2]),
            Rgba(1, 0, 0, 0.3),
        )
        self.meshcat.SetTransform("closing_region", RigidTransform(box_center))

    def compute_candidate_grasps(self, pcd, context, candidate_num=3, random_seed=5):
        x_min = -0.03
        x_max = 0.03
        phi_min = -np.pi / 3
        phi_max = np.pi / 3
        np.random.seed(random_seed)

        ball_radius = 0.002

        candidate_count = 0
        candidate_lst = []

        # Downsample the point cloud to reduce computation
        sample_indices = np.random.choice(pcd.size(), size=min(100, pcd.size()), replace=False)
        sampled_points = pcd.xyzs()[:, sample_indices]

        kdtree = KDTree(sampled_points.T)

        while candidate_count < candidate_num:
            index = np.random.randint(0, sampled_points.shape[1])
            X_WF = self.compute_darboux_frame(index, sample_indices, pcd, kdtree, ball_radius)
            if X_WF is None:
                continue
            x = np.random.rand() * (x_max - x_min) + x_min
            phi = np.random.rand() * (phi_max - phi_min) + phi_min
            X_FT = RigidTransform(RotationMatrix.MakeZRotation(phi), [x, 0, 0])
            X_WT = X_WF.multiply(X_FT)

            signed_distance, X_WG = self.find_minimum_distance(sampled_points, X_WT, context)
            if np.isnan(signed_distance) or X_WG is None:
                continue
            if not self.check_collision(sampled_points, X_WG):
                continue

            if not self.check_nonempty(sampled_points, X_WG):
                continue

            candidate_lst.append(X_WG)
            candidate_count += 1

        return candidate_lst

    def compute_sdf(self, sampled_points, context, X_G, visualize=False):
        plant, scene_graph = self.plant, self.scene_graph
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(plant, context)
        scene_graph_context = self.diagram.GetMutableSubsystemContext(scene_graph, context)
        # plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)

        # if visualize:
        #     self.meshcat.Delete()
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        # Downsample the point cloud to reduce computation
        # sample_indices = np.random.choice(pcd.size(), size=min(100, pcd.size()), replace=False)

        pcd_sdf = np.inf
        for pt in sampled_points.T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < pcd_sdf:
                    pcd_sdf = distance

        return pcd_sdf

    def set_diagram(self, diagram):
        self.diagram = diagram

    def check_collision(self, sampled_points, X_G, visualize=False):
        sdf = self.compute_sdf(sampled_points, X_G, visualize)
        return sdf > 0

    def run_grasp(self, pcd, context, visualize=False):
        # pcd = PointCloud(points.shape[0])
        # pcd.mutable_xyzs()[:] = points.T
        # #pcd.mutable_rgbs()[:] = colors.T

        candidate_grasps = self.compute_candidate_grasps(pcd, context, candidate_num=5)

        if not candidate_grasps:
            print("No valid grasps found.")
            return None
        best_grasp = candidate_grasps[0]
        g = candidate_grasps[0]
        # for ix, g in enumerate(candidate_grasps):
        #     AddMeshcatTriad(self.meshcat, f"grasp_pose_{ix}", length=0.1, radius=0.005, X_PT=g)
        for g in candidate_grasps:
            # Realign the gripper with the z-axis
            R = g.rotation().matrix()
            gripper_x_axis = R[:, 0]
            gripper_y_axis = R[:, 1]

            # Compute rotation matrix to align gripper x-axis with the positive z-direction
            target_axis = np.array([0.0, 0.0, -1.0])  # Desired x-axis direction
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
                g.set_rotation(RotationMatrix(R_align @ R))

            # Recalculate axes after rotation
            R = g.rotation().matrix()
            gripper_x_axis = R[:, 0]
            gripper_y_axis = R[:, 1]

            # Check alignment with thresholds
            x_alignment = np.dot(gripper_x_axis, target_axis)
            y_alignment = np.dot(gripper_y_axis, np.array([1.0, 0.0, 0.0]))

            if x_alignment > 0.9 and y_alignment > 0.9:  # Adjust thresholds if needed
                best_grasp = g
                break

        if visualize:
            self.draw_grasp_candidate(best_grasp, refresh=True)

        return best_grasp
