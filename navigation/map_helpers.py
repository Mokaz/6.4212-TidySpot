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
from typing import List, Tuple

class PointCloudProcessor(LeafSystem):
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
        self.DeclareAbstractOutputPort("grid_map", lambda: AbstractValue.Make(np.full((100, 100), -1)), self.CalcGridMap)
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.grid_map = np.full((100, 100), -1)  # Initialized with a fixed size grid map with unexplored (-1)
        self.area_graph = {}  # Initialize an empty graph to maintain new area information
        self.height_threshold = height_threshold  # Threshold to differentiate between free space and obstacles

    def CalcGridMap(self, context: Context, output: AbstractValue):
        for camera_name in self._camera_names:
            point_cloud = self._point_cloud_inputs[camera_name].Eval(context).xyzs()  # Corrected to xyzs()
            valid_points = point_cloud[:, np.isfinite(point_cloud).all(axis=0)]  # Filter points that are finite
            ox, oy, free_ox, free_oy = self.pointcloud_to_grid(valid_points)  # Convert to grid
            self.grid_map = self.update_grid_map(self.grid_map, ox, oy, free_ox, free_oy)
            self.update_area_graph(ox, oy)  # Update the area graph with new obstacles

        self.mark_robot_footprint_as_free()
        self.update_grid_map_from_area_graph()  # Update grid_map using area graph
        output.set_value(self.grid_map)

    def connect_point_clouds(self, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            point_cloud_output = self._point_clouds[camera_name].GetOutputPort("point_cloud")
            point_cloud_input = self._point_cloud_inputs[camera_name]

            builder.Connect(point_cloud_output, point_cloud_input)

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

    def update_grid_map(self, grid_map: np.ndarray, ox: List[int], oy: List[int], free_ox: List[int], free_oy: List[int]) -> np.ndarray:
        """
        Updates the grid map with obstacles and free space from the point cloud.

        grid_map: Current grid map to be updated
        ox, oy: Lists of grid coordinates representing obstacles
        free_ox, free_oy: Lists of grid coordinates representing free space
        Returns:
            Updated grid map
        """
        for x, y in zip(ox, oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                grid_map[x, y] = 1  # Mark cell as occupied (obstacle)

        for x, y in zip(free_ox, free_oy):
            if 0 <= x < grid_map.shape[0] and 0 <= y < grid_map.shape[1]:
                # Only mark as free if it has not been marked as an obstacle
                if grid_map[x, y] == -1:  # Unexplored space
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
        """
        center_x, center_y = self.grid_map.shape[0] // 2, self.grid_map.shape[1] // 2
        radius_in_cells = int(self.robot_radius / self.resolution)
        for dx in range(-radius_in_cells, radius_in_cells + 1):
            for dy in range(-radius_in_cells, radius_in_cells + 1):
                if dx ** 2 + dy ** 2 <= radius_in_cells ** 2:  # Check if within robot radius
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.grid_map.shape[0] and 0 <= y < self.grid_map.shape[1]:
                        if self.grid_map[x, y] == -1:  # If unexplored, mark as free
                            self.grid_map[x, y] = 0

    def visualize_grid_map(self):
        """
        Visualizes the current grid map using matplotlib.
        """
        plt.figure(figsize=(10, 10))
        cmap = plt.get_cmap('Greys', 3)  # Define a colormap with 3 levels (unexplored, free, obstacle)
        plt.imshow(self.grid_map.T, cmap=cmap, origin='lower', extent=[-self.grid_map.shape[1] * self.resolution / 2, self.grid_map.shape[1] * self.resolution / 2, -self.grid_map.shape[0] * self.resolution / 2, self.grid_map.shape[0] * self.resolution / 2], vmin=-1, vmax=1)
        plt.title("Obstacle Grid Map")
        plt.xlabel("X-axis (meters)")
        plt.ylabel("Y-axis (meters)")
        plt.colorbar(ticks=[-1, 0, 1], label="Cell State (-1: Unexplored, 0: Free, 1: Obstacle)")
        plt.grid(True)
        plt.show()

