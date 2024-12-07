import numpy as np
import math
import matplotlib.pyplot as plt
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    LeafSystem,
    Context,
    Diagram,
    PointCloud,
    RigidTransform,
    RotationMatrix
)
from typing import List, Tuple, Dict
from scipy.ndimage import label
from utils import (
    add_sphere_to_meshcat_xy_plane,
    convert_to_grid_coordinates,
    convert_to_world_coordinates,
)

ADD_DETECTIONS_TO_GRIDMAP = True
VISUALIZE_GRID_MAP = False


class PointCloudMapper(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], point_clouds, resolution, robot_length=1.3, robot_width=0.7, height_threshold=0.03, meshcat=None):
        LeafSystem.__init__(self)
        self._point_clouds = point_clouds
        self._camera_names = camera_names
        # remove the back camera
        self._camera_names.remove("back")
        self._cameras = {
            point_cloud_name: station.GetSubsystemByName(f"rgbd_sensor_{point_cloud_name}") for point_cloud_name in camera_names
        }
        self.meshcat = meshcat

        self.DeclareVectorInputPort("spot.state_estimated",20)
        # Input ports
        self._point_cloud_inputs = {
            point_cloud_name: self.DeclareAbstractInputPort(f"{point_cloud_name}.point_cloud", AbstractValue.Make(PointCloud())) for point_cloud_name in camera_names
        }
        self._object_pcd_input = self.DeclareAbstractInputPort(f"object_detection_cropped_point_clouds", AbstractValue.Make(PointCloud()))

        # Output ports
        self.DeclareAbstractOutputPort("grid_map", lambda: AbstractValue.Make(np.full((100, 100), -1)), self.CalcGridMap)
        self.DeclareAbstractOutputPort("object_clusters", lambda: AbstractValue.Make({}), self.CalcObjectClusters)
        self.DeclareVectorOutputPort("frontier", 2, self.CalcFrontier)

        self.resolution = resolution
        self.grid_map = np.full((100, 100), -1)  # Initialized with a fixed size grid map with unexplored (-1)
        self.height_threshold = height_threshold  # Threshold to differentiate between free space and obstacles
        self.object_clusters = {}  # Dictionary to hold obstacle clusters
        self.robot_grid_pos = (50, 50)  # Initialize robot position in grid coordinates
        self.robot_theta = 0  # Initialize robot orientation
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.frontier_strategy = "closest"
        self.mark_robot_footprint_as_free()

    def CalcObjectClusters(self, context: Context, output: AbstractValue):
        # Set the output to the current object clusters
        output.set_value(self.object_clusters)

    def CalcGridMap(self, context: Context, output: AbstractValue):
        self.robot_state = self.GetInputPort("spot.state_estimated").Eval(context)
        self.robot_grid_pos = convert_to_grid_coordinates(self.robot_state[0], self.robot_state[1], self.resolution, self.grid_map.shape) # convert robot position to grid coordinates
        self.robot_theta = self.robot_state[2] # get robot orientation

        # Process regular point clouds to mark obstacles
        for camera_name in self._camera_names:
            point_cloud = self._point_cloud_inputs[camera_name].Eval(context).xyzs()  # Corrected to xyzs()
            valid_points = point_cloud[:, np.isfinite(point_cloud).all(axis=0)]  # Filter points that are finite
            ox, oy, free_ox, free_oy = self.pointcloud_to_grid(valid_points)  # Convert to grid
            self.grid_map = self.update_grid_map(self.grid_map, ox, oy, free_ox, free_oy, value=1)  # Mark obstacles with value 1

        if ADD_DETECTIONS_TO_GRIDMAP:
            # Process object point cloud to mark objects
            object_pcd = self._object_pcd_input.Eval(context)

            # self.meshcat.SetObject("object_point_cloud", object_pcd)

            object_point_cloud_points = object_pcd.xyzs()
            valid_object_points = object_point_cloud_points[:, np.isfinite(object_point_cloud_points).all(axis=0)]  # Filter points that are finite

            ox, oy = self.object_pointcloud_to_grid(valid_object_points)  # Convert to grid (objects only)
            self.grid_map = self.update_grid_map(self.grid_map, ox, oy, [], [], value=2)  # Mark objects with value 2

        if VISUALIZE_GRID_MAP:
            self.visualize_grid_map()  # Visualize the grid map
        self.cluster_objects()  # Cluster objects in the grid map
        # print(f"Object clusters: {self.object_clusters}")

        output.set_value(self.grid_map)

    def CalcFrontier(self, context: Context, output):
        # Set the output to the current object clusters
        frontiers = self.find_unexplored_frontiers()
        frontiers = self.cluster_frontiers(frontiers)
        if len(frontiers) == 0:
            output.SetFromVector([0, 0])
            return
        else:
            if self.frontier_strategy == "random":
                frontier = self.pick_random_frontier(frontiers)
            else:
                frontier = self.pick_closest_frontier(frontiers)
        if frontier is None:
            output.SetFromVector([0, 0])
            return
        frontier_w = convert_to_world_coordinates(frontier[0], frontier[1], self.resolution, self.grid_map.shape)
        output.SetFromVector(frontier_w)

    def connect_components(self, point_cloud_cropper, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            point_cloud_output = self._point_clouds[camera_name].GetOutputPort("point_cloud")
            point_cloud_input = self._point_cloud_inputs[camera_name]
            # label_image_output = self._cameras[camera_name].GetOutputPort("label_image")
            # label_image_input = self._label_image_inputs[camera_name]

            builder.Connect(point_cloud_output, point_cloud_input)
            # builder.Connect(label_image_output, label_image_input)

        builder.Connect(point_cloud_cropper.GetOutputPort("object_detection_cropped_point_clouds"), self._object_pcd_input)

        builder.Connect(
            station.GetOutputPort("spot.state_estimated"),
            self.GetInputPort("spot.state_estimated"),
        )



    def object_pointcloud_to_grid(self, point_cloud: np.ndarray) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Converts a point cloud to grid coordinates.

        point_cloud: 3xN numpy array of points (x, y, z)
        Returns:
            ox, oy: Lists of grid coordinates representing obstacles
            free_ox, free_oy: Lists of grid coordinates representing free space
        """
        # Initialize empty lists for objects
        ox, oy = [], []

        for i in range(point_cloud.shape[1]):
            x, y = point_cloud[0, i], point_cloud[1, i]
            ix, iy = convert_to_grid_coordinates(x, y, self.resolution, self.grid_map.shape)
            ox.append(ix)
            oy.append(iy)

        return ox, oy

    def pointcloud_to_grid(self, point_cloud: np.ndarray) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Converts a point cloud to grid coordinates.

        point_cloud: 3xN numpy array of points (x, y, z)
        Returns:
            ox, oy: Lists of grid coordinates representing obstacles
            free_ox, free_oy: Lists of grid coordinates representing free space
        """
        # Extract points in the x, y plane and classify based on z
        obstacle_points = point_cloud[:, point_cloud[2, :] >= self.height_threshold]  # Points above threshold are obstacles
        free_points = point_cloud[:, point_cloud[2, :] < self.height_threshold]  # Points below threshold are free space

        # Initialize empty lists for obstacle and free space coordinates
        ox, oy, free_ox, free_oy = [], [], [], []

        for i in range(obstacle_points.shape[1]):
            x, y = obstacle_points[0, i], obstacle_points[1, i]
            ix, iy = convert_to_grid_coordinates(x, y, self.resolution, self.grid_map.shape)
            ox.append(ix)
            oy.append(iy)

        for i in range(free_points.shape[1]):
            x, y = free_points[0, i], free_points[1, i]
            ix, iy = convert_to_grid_coordinates(x, y, self.resolution, self.grid_map.shape)
            free_ox.append(ix)
            free_oy.append(iy)

        return ox, oy, free_ox, free_oy

    def update_grid_map(self, grid_map: np.ndarray, ox: List[int], oy: List[int], free_ox: List[int], free_oy: List[int], value: int) -> np.ndarray:
        """
        Updates the grid map with obstacles, objects, and free space from the point cloud.

        grid_map: Current grid map to be updated
        ox, oy: Lists of grid coordinates representing obstacles or objects
        free_ox, free_oy: Lists of grid coordinates representing free space
        value: Value to assign to the obstacle or object cells (1 for obstacles, 2 for objects)
        Returns:
            Updated grid map
        """
        for x, y in zip(free_ox, free_oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                # Mark as free even if it was previously marked as an obstacle, to reflect dynamic changes
                grid_map[x, y] = 0  # Mark cell as free space
        for x, y in zip(ox, oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                grid_map[x, y] = value  # Mark cell as occupied (obstacle or object)

        return grid_map

    def mark_robot_footprint_as_free(self):
        """
        Marks the grid cells corresponding to the robot's footprint as free space.
        """
        center_x, center_y = self.robot_grid_pos
        half_length_cells = int(self.robot_length / (2 * self.resolution))
        half_width_cells = int(self.robot_width / (2 * self.resolution))

        for dx in range(-half_length_cells, half_length_cells + 1):
            for dy in range(-half_width_cells, half_width_cells + 1):
                # Rotate the point based on the robot's orientation
                rotated_x = dx * math.cos(self.robot_theta) - dy * math.sin(self.robot_theta)
                rotated_y = dx * math.sin(self.robot_theta) + dy * math.cos(self.robot_theta)
                x = int(center_x + rotated_x)
                y = int(center_y + rotated_y)
                if 0 <= x < self.grid_map.shape[0] and 0 <= y < self.grid_map.shape[1]:
                    self.grid_map[x, y] = 0  # Mark as free, regardless of previous state

    def cluster_objects(self):
        """
        Clusters objects together if they are in the same blob using connected-component labeling.
        """
        # Label connected components in the object grid (value == 2)
        labeled_grid, num_features = label(self.grid_map == 2)

        # Create a dictionary to store clusters
        self.object_clusters = {}
        for i in range(1, num_features + 1):
            cluster_indices = np.argwhere(labeled_grid == i)
            grid_points = cluster_indices.tolist()

            # TODO: Add this in if we are getting incorrect pointclouds
            # if len(grid_points) < 2:
            #     # not enough points to register as an object?
            #     continue

            # Calculate a good centroid that is part of the cluster
            centroid_index = len(cluster_indices) // 2  # Take the middle point in the sorted list as a centroid
            centroid_grid = cluster_indices[centroid_index]

            # Convert the centroid from grid to world coordinates
            centroid_x = (centroid_grid[0] - (self.grid_map.shape[0] // 2)) * self.resolution
            centroid_y = (centroid_grid[1] - (self.grid_map.shape[1] // 2)) * self.resolution

            self.object_clusters[i] = {
                "grid_points": grid_points,
                "centroid": {
                    "grid": centroid_grid.tolist(),
                    "world": (centroid_x, centroid_y)
                }
            }

    def find_unexplored_frontiers(self) -> List[Tuple[int, int]]:
        """
        Finds unexplored frontier cells in the grid map. A frontier cell is a free cell that is adjacent to at least one unexplored cell.

        Returns:
            List of tuples representing the grid coordinates of frontier cells.
        """
        frontiers = []
        rows, cols = self.grid_map.shape

        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if self.grid_map[x, y] == 0:  # Free space
                    neighbors = [
                        self.grid_map[x - 1, y],  # Up
                        self.grid_map[x + 1, y],  # Down
                        self.grid_map[x, y - 1],  # Left
                        self.grid_map[x, y + 1]   # Right
                    ]
                    if -1 in neighbors and 1 not in neighbors:  # Check for unexplored neighbor and not next to obstacle
                        frontiers.append((x, y))
        return frontiers

    def cluster_frontiers(self, frontiers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Clusters frontier cells and returns the central points of each cluster, discarding clusters that are too small.

        frontiers: List of frontier cells (grid coordinates).
        Returns:
            List of tuples representing the central points of each cluster.
        """
        frontier_grid = np.zeros_like(self.grid_map, dtype=int)

        # Mark frontier cells in a separate grid
        for x, y in frontiers:
            frontier_grid[x, y] = 1

        # Label connected components of frontier cells
        labeled_grid, num_features = label(frontier_grid)

        central_points = []
        for i in range(1, num_features + 1):
            cluster_indices = np.argwhere(labeled_grid == i)

            # Discard clusters that are too small
            if len(cluster_indices) < 5:  # Adjust the threshold as needed
                continue

            # Calculate the centroid
            centroid_index = len(cluster_indices) // 2
            central_point = tuple(cluster_indices[centroid_index])
            central_points.append(central_point)

        return central_points

    def pick_closest_frontier(self, frontiers: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Picks the closest frontier cell to the robot's current position.

        frontiers: List of tuples representing the grid coordinates of frontier cells.
        Returns:
            Tuple representing the grid coordinates of the closest frontier cell.
        """
        # has to be at least a meter away
        min_dist = float('inf')
        closest_frontier = None
        for frontier in frontiers:
            dist = math.sqrt((frontier[0] - self.robot_grid_pos[0]) ** 2 + (frontier[1] - self.robot_grid_pos[1]) ** 2)
            if dist < min_dist and dist > 10:
                min_dist = dist
                closest_frontier = frontier

        return closest_frontier

    def pick_random_frontier(self, frontiers: List[Tuple[int, int]]) -> Tuple[int, int]:
        random_frontier = frontiers[np.random.randint(0, len(frontiers))]
        return random_frontier

    def visualize_grid_map(self):
        """
        Visualizes the current grid map using matplotlib.
        """
        plt.figure(figsize=(10, 10))
        cmap = plt.get_cmap('Greys', 4)  # Define a colormap with 4 levels (unexplored, free, obstacle, object)
        plt.imshow(self.grid_map.T, cmap=cmap, origin='lower', extent=[-self.grid_map.shape[1] * self.resolution / 2, self.grid_map.shape[1] * self.resolution / 2, -self.grid_map.shape[0] * self.resolution / 2, self.grid_map.shape[0] * self.resolution / 2], vmin=-1, vmax=2)
        plt.title("Grid Map with Obstacles and Objects")
        plt.xlabel("X-axis (meters)")
        plt.ylabel("Y-axis (meters)")
        plt.colorbar(ticks=[-1, 0, 1, 2], label="Cell State (-1: Unexplored, 0: Free, 1: Obstacle, 2: Object)")
        plt.grid(True)
        plt.show()
