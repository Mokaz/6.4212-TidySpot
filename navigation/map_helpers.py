import numpy as np
import math
import matplotlib.pyplot as plt
import heapq
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    LeafSystem,
    Context,
    ImageRgba8U,
    ImageDepth32F,
    ImageLabel16I,
    Diagram,
    PointCloud,
)
from typing import List, Tuple

class PointCloudProcessor(LeafSystem):
    def __init__(self, station:Diagram, camera_names: List[str], point_clouds, resolution, robot_radius):
        LeafSystem.__init__(self)
        self._point_clouds = point_clouds
        self._camera_names = camera_names
        self._cameras = {
            point_cloud_name: station.GetSubsystemByName(f"rgbd_sensor_{point_cloud_name}") for point_cloud_name in camera_names
        }
        self._point_cloud_inputs = {
            point_cloud_name: self.DeclareAbstractInputPort( f"{point_cloud_name}.point_cloud", AbstractValue.Make(PointCloud()) ) for point_cloud_name in camera_names
        }
        self.DeclareAbstractOutputPort("grid_map", lambda: AbstractValue.Make(np.zeros((1, 1))), self.CalcGridMap)
        self.resolution = resolution
        self.robot_radius = robot_radius

    def CalcGridMap(self, context: Context, output: AbstractValue):
        grid_map = np.zeros((0, 0))
        for camera_name in self._cameras:
            point_cloud = self.EvalAbstractInput(context, self._point_cloud_inputs[camera_name]).data()
            ox, oy = self.pointcloud_to_grid(point_cloud)
            grid_map = self.update_grid_map(grid_map, ox, oy)
        output.set_value(grid_map)

    def connect_point_clouds(self, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            point_cloud_output = self._point_clouds[camera_name].GetOutputPort("point_cloud")
            point_cloud_input = self._point_cloud_inputs[camera_name]

            builder.Connect(point_cloud_output, point_cloud_input)

    def pointcloud_to_grid(self, point_cloud):
        """
        Converts a point cloud to a grid map.

        point_cloud: Nx3 numpy array of points (x, y, z)
        """
        # Extract points in the x, y plane (filter out points based on z to remove floor/ceiling noise)
        point_cloud_2d = point_cloud[point_cloud[:, 2] < 0.5]  # Filter points above 0.5m to ignore non-floor objects

        # Initialize empty lists for obstacle coordinates
        ox, oy = [], []

        for point in point_cloud_2d:
            x, y, _ = point
            ix = round(x / self.resolution)
            iy = round(y / self.resolution)
            ox.append(ix)
            oy.append(iy)

        return ox, oy

    def update_obstacles(self, planner, point_cloud):
        """
        Updates the obstacles in the planner using a point cloud.
        """
        ox, oy = self.pointcloud_to_grid(point_cloud)
        planner.update_with_new_area({(x, y): 1 for x, y in zip(ox, oy)})
