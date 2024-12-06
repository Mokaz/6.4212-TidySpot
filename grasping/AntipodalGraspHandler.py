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

    def compute_darboux_frame(self, index, pcd, kdtree, ball_radius=0.002, max_nn=50):
        points = pcd.xyzs()
        normals = pcd.normals()

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

    def find_minimum_distance(self, pcd, X_WG):
        y_grid = np.linspace(-0.05, 0.05, 10)

        max_distance = 0
        X_WGnew = None

        for delta in y_grid:
            X_WGsearch = X_WG.multiply(RigidTransform([0, delta, 0]))
            distance = self.compute_sdf(pcd, X_WGsearch)
            if distance >= 0:
                max_distance = distance
                X_WGnew = X_WGsearch
            elif distance < 0:
                break

        if X_WGnew is None:
            return np.nan, None
        return max_distance, X_WGnew

    def check_nonempty(self, pcd, X_WG, visualize=False):
        pcd_W_np = pcd.xyzs()
        crop_min = [-0.05, 0.1, -0.00625]
        crop_max = [0.05, 0.1125, 0.00625]

        X_GW = X_WG.inverse()
        pcd_G_np = X_GW.multiply(pcd_W_np)

        indices = np.all(
            (
                crop_min[0] <= pcd_G_np[0, :],
                pcd_G_np[0, :] <= crop_max[0],
                crop_min[1] <= pcd_G_np[1, :],
                pcd_G_np[1, :] <= crop_max[1],
                crop_min[2] <= pcd_G_np[2, :],
                pcd_G_np[2, :] <= crop_max[2],
            ),
            axis=0,
        )

        is_nonempty = indices.any()

        if visualize:
            self.meshcat.Delete()
            pcd_G = PointCloud(pcd)
            pcd_G.mutable_xyzs()[:] = pcd_G_np

            self.draw_grasp_candidate(RigidTransform())
            self.meshcat.SetObject("cloud", pcd_G)

            box_length = np.array(crop_max) - np.array(crop_min)
            box_center = (np.array(crop_max) + np.array(crop_min)) / 2.0
            self.meshcat.SetObject(
                "closing_region",
                Box(box_length[0], box_length[1], box_length[2]),
                Rgba(1, 0, 0, 0.3),
            )
            self.meshcat.SetTransform("closing_region", RigidTransform(box_center))

        return is_nonempty

    def compute_candidate_grasps(self, pcd, candidate_num=10, random_seed=5):
        x_min = -0.03
        x_max = 0.03
        phi_min = -np.pi / 3
        phi_max = np.pi / 3
        np.random.seed(random_seed)

        kdtree = KDTree(pcd.xyzs().T)
        ball_radius = 0.002

        candidate_count = 0
        candidate_lst = []

        while candidate_count < candidate_num:
            index = np.random.randint(0, pcd.size())
            X_WF = self.compute_darboux_frame(index, pcd, kdtree, ball_radius)
            if X_WF is None:
                continue
            x = np.random.rand() * (x_max - x_min) + x_min
            phi = np.random.rand() * (phi_max - phi_min) + phi_min
            X_FT = RigidTransform(RotationMatrix.MakeZRotation(phi), [x, 0, 0])
            X_WT = X_WF.multiply(X_FT)

            signed_distance, X_WG = self.find_minimum_distance(pcd, X_WT)
            if np.isnan(signed_distance) or X_WG is None:
                continue
            if not self.check_collision(pcd, X_WG):
                continue

            if not self.check_nonempty(pcd, X_WG):
                continue

            candidate_lst.append(X_WG)
            candidate_count += 1

        return candidate_lst

    def compute_sdf(self, pcd, X_G, visualize=False):
        plant, scene_graph = self.plant, self.scene_graph
        diagram = self.plant.GetParentDiagram()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_G)

        if visualize:
            diagram.ForcedPublish(context)

        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)

        pcd_sdf = np.inf
        for pt in pcd.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for body_index in range(len(distances)):
                distance = distances[body_index].distance
                if distance < pcd_sdf:
                    pcd_sdf = distance

        return pcd_sdf

    def check_collision(self, pcd, X_G, visualize=False):
        sdf = self.compute_sdf(pcd, X_G, visualize)
        return sdf > 0

    def run_grasp(self, points, colors, lims=None, flip_before_calc=True, visualize=False):
        pcd = PointCloud(points.shape[0])
        pcd.mutable_xyzs()[:] = points.T
        pcd.mutable_rgbs()[:] = colors.T

        candidate_grasps = self.compute_candidate_grasps(pcd, candidate_num=10)

        if not candidate_grasps:
            print("No valid grasps found.")
            return None

        best_grasp = candidate_grasps[0]
        for grasp in candidate_grasps:
            _, grasp_candidate = self.find_minimum_distance(pcd, grasp)
            if grasp_candidate is not None:
                best_grasp = grasp_candidate

        if visualize:
            self.draw_grasp_candidate(best_grasp, refresh=True)

        return best_grasp
