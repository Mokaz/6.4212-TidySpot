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

class PointCloudMapper(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], point_clouds, resolution, robot_radius, height_threshold=0.1):
        LeafSystem.__init__(self)
        self._point_clouds = point_clouds
        self._camera_names = camera_names
        self._cameras = {
            point_cloud_name: station.GetSubsystemByName(f"rgbd_sensor_{point_cloud_name}") for point_cloud_name in camera_names
        }
        self._point_cloud_inputs = {
            point_cloud_name: self.DeclareAbstractInputPort(f"{point_cloud_name}.point_cloud", AbstractValue.Make(PointCloud())) for point_cloud_name in camera_names
        }
        self._object_point_cloud = self.DeclareAbstractInputPort(f"cropped_point_cloud", AbstractValue.Make(PointCloud()))
        self.DeclareAbstractOutputPort("grid_map", lambda: AbstractValue.Make(np.full((100, 100), -1)), self.CalcGridMap)
        self.DeclareAbstractOutputPort("object_clusters", lambda: AbstractValue.Make({}), self.CalcObjectClusters)
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.grid_map = np.full((100, 100), -1)  # Initialized with a fixed size grid map with unexplored (-1)
        self.area_graph = {}  # Initialize an empty graph to maintain new area information
        self.height_threshold = height_threshold  # Threshold to differentiate between free space and obstacles
        self.object_clusters = {}  # Dictionary to hold obstacle clusters

    def CalcObjectClusters(self, context: Context, output: AbstractValue):
        # Set the output to the current object clusters
        output.set_value(self.object_clusters)

    def CalcGridMap(self, context: Context, output: AbstractValue):
        # Reset grid map to unexplored at the start of each update
        self.grid_map.fill(-1)

        # Process regular point clouds to mark obstacles
        for camera_name in self._camera_names:
            point_cloud = self._point_cloud_inputs[camera_name].Eval(context).xyzs()  # Corrected to xyzs()
            valid_points = point_cloud[:, np.isfinite(point_cloud).all(axis=0)]  # Filter points that are finite
            ox, oy, free_ox, free_oy = self.pointcloud_to_grid(valid_points)  # Convert to grid
            self.grid_map = self.update_grid_map(self.grid_map, ox, oy, free_ox, free_oy, value=1)  # Mark obstacles with value 1
            self.update_area_graph(ox, oy)  # Update the area graph with new obstacles

        # Process object point cloud to mark objects
        object_point_cloud = self._object_point_cloud.Eval(context).xyzs()
        valid_object_points = object_point_cloud[:, np.isfinite(object_point_cloud).all(axis=0)]  # Filter points that are finite
        ox, oy, _, _ = self.pointcloud_to_grid(valid_object_points)  # Convert to grid (objects only)
        self.grid_map = self.update_grid_map(self.grid_map, ox, oy, [], [], value=2)  # Mark objects with value 2

        self.mark_robot_footprint_as_free()
        self.update_grid_map_from_area_graph()  # Update grid_map using area graph
        self.visualize_grid_map()  # Visualize the grid map
        self.cluster_objects()  # Cluster objects in the grid map
        output.set_value(self.grid_map)

    def connect_point_clouds(self, point_cloud_cropper, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            point_cloud_output = self._point_clouds[camera_name].GetOutputPort("point_cloud")
            point_cloud_input = self._point_cloud_inputs[camera_name]
            # label_image_output = self._cameras[camera_name].GetOutputPort("label_image")
            # label_image_input = self._label_image_inputs[camera_name]

            builder.Connect(point_cloud_output, point_cloud_input)
            # builder.Connect(label_image_output, label_image_input)

        builder.Connect(point_cloud_cropper.GetOutputPort("cropped_point_cloud"), self._object_point_cloud)

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
            ix = int(round(x / self.resolution)) + (self.grid_map.shape[0] // 2)  # Offset to center grid
            iy = int(round(y / self.resolution)) + (self.grid_map.shape[1] // 2)  # Offset to center grid
            ox.append(ix)
            oy.append(iy)

        for i in range(free_points.shape[1]):
            x, y = free_points[0, i], free_points[1, i]
            ix = int(round(x / self.resolution)) + (self.grid_map.shape[0] // 2)  # Offset to center grid
            iy = int(round(y / self.resolution)) + (self.grid_map.shape[1] // 2)  # Offset to center grid
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
        for x, y in zip(ox, oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                grid_map[x, y] = value  # Mark cell as occupied (obstacle or object)

        for x, y in zip(free_ox, free_oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                # Mark as free even if it was previously marked as an obstacle, to reflect dynamic changes
                grid_map[x, y] = 0  # Mark cell as free space
        return grid_map

    def update_area_graph(self, ox: List[int], oy: List[int]):
        """
        Updates the area graph with new obstacles.

        ox, oy: Lists of grid coordinates representing obstacles
        """
        for x, y in zip(ox, oy):
            if (x, y) not in self.area_graph:
                self.area_graph[(x, y)] = 1  # Add new obstacle node
            else:
                self.area_graph[(x, y)] += 1  # Increment obstacle count for existing node

    def update_grid_map_from_area_graph(self):
        """
        Updates the grid map using the information stored in the area graph.
        """
        for (x, y), count in self.area_graph.items():
            if 0 <= x < self.grid_map.shape[0] and 0 <= y < self.grid_map.shape[1]:
                self.grid_map[x, y] = 1  # Mark cell as occupied based on area graph information

    def mark_robot_footprint_as_free(self):
        """
        Marks the grid cells corresponding to the robot's footprint as free space.

        # TODO: make this the actual footprint of the robot
        """
        center_x, center_y = self.grid_map.shape[0] // 2, self.grid_map.shape[1] // 2
        radius_in_cells = int(self.robot_radius / self.resolution)
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                if dx ** 2 + dy ** 2 <= radius_in_cells ** 2:  # Check if within robot radius
                    x, y = center_x + dx, center_y + dy
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