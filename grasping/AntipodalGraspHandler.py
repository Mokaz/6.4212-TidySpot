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
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

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

    def compute_darboux_frame_batch(self, points, normals, kdtree, indices, ball_radius=0.002, max_nn=50):
        frames = []
        for index in indices:
            query = points[:, index]
            distances, neighbors = kdtree.query(query, k=max_nn, distance_upper_bound=ball_radius)
            valid_neighbors = neighbors[np.isfinite(distances)]

            neighbor_normals = normals[:, valid_neighbors]
            N = neighbor_normals @ neighbor_normals.T
            eigenvalues, eigenvectors = np.linalg.eigh(N)
            V = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

            v_1 = V[:, 0] if V[:, 0].dot(normals[:, index]) < 0 else -V[:, 0]
            R = np.column_stack((V[:, 1], v_1, V[:, 2]))
            R = R @ np.diag([1, 1, np.linalg.det(R)])
            frames.append(RigidTransform(RotationMatrix(R), query))
        return frames

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

        X_GW = X_WG.inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)

        within_bounds = np.logical_and(
            np.all(pcd_G_np >= crop_min[:, None], axis=0),
            np.all(pcd_G_np <= crop_max[:, None], axis=0)
        )

        is_nonempty = np.any(within_bounds)
        return is_nonempty

    def compute_candidate_grasps(self, pcd, context, candidate_num=3, random_seed=5):
        x_min = -0.03
        x_max = 0.03
        phi_min = -np.pi / 3
        phi_max = np.pi / 3
        np.random.seed(random_seed)

        ball_radius = 0.002
        candidate_count = 0
        candidate_lst = []

        downsampled_indices = np.random.choice(pcd.size(), size=min(100, pcd.size()), replace=False)
        sampled_points = pcd.xyzs()[:, downsampled_indices]
        kdtree = KDTree(sampled_points.T)

        pcd.EstimateNormals(radius=0.01, num_closest=30)
        normals = pcd.normals()[:,downsampled_indices]

        while candidate_count < candidate_num:
            indices = np.random.choice(sampled_points.shape[1], size=candidate_num, replace=False)
            X_WF_list = self.compute_darboux_frame_batch(sampled_points, normals, kdtree, indices, ball_radius)

            for X_WF in X_WF_list:
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
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        distances = np.array([
            min([dist.distance for dist in query_object.ComputeSignedDistanceToPoint(pt)])
            for pt in sampled_points.T
        ])

        return np.min(distances)

    def set_diagram(self, diagram):
        self.diagram = diagram

    def check_collision(self, sampled_points, X_G, visualize=False):
        sdf = self.compute_sdf(sampled_points, X_G, visualize)
        return sdf > 0

    def run_grasp(self, pcd, context, visualize=False):
        candidate_grasps = self.compute_candidate_grasps(pcd, context, candidate_num=1)

        if not candidate_grasps:
            print("No valid grasps found.")
            return None

        best_grasp = None

        for g in candidate_grasps:
            R = g.rotation().matrix()
            gripper_x_axis = R[:, 0]
            target_axis = np.array([0.0, 0.0, -1.0])
            cross_prod = np.cross(gripper_x_axis, target_axis)
            dot_prod = np.dot(gripper_x_axis, target_axis)

            if np.linalg.norm(cross_prod) > 1e-6:
                axis = cross_prod / np.linalg.norm(cross_prod)
                angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))

                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                g.set_rotation(RotationMatrix(R_align @ R))

            R = g.rotation().matrix()
            gripper_x_axis = R[:, 0]
            # x_alignment = np.dot(gripper_x_axis, target_axis)

            best_grasp = g

        if visualize:
            self.draw_grasp_candidate(best_grasp, refresh=True)

        return best_grasp
