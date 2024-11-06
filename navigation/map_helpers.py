import numpy as np
import math
import matplotlib.pyplot as plt
import heapq


class PointCloudProcessor:
    def __init__(self, resolution, robot_radius):
        self.resolution = resolution
        self.robot_radius = robot_radius

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
