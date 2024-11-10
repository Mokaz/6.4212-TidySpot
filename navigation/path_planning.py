"""
This class maintains the map of the world and allows for A* grid planning

original author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

adapted by Matthew Trang (trangml) to allow for updating the grid and changing the start and goal locations

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
and PythonRobotics implementation (https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py)
"""

import math
import numpy as np
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
from navigation.map_helpers import PointCloudProcessor

show_animation = True

class DynamicPathPlanner(LeafSystem):
    def __init__(self, station: Diagram, point_cloud_processor, ox, oy, resolution, robot_radius):
        """
        Initialize grid map for A* planning.

        ox: x positions of obstacles
        oy: y positions of obstacles
        resolution: grid resolution
        robot_radius: robot radius
        """
        LeafSystem.__init__(self)
        self.point_cloud_processor = point_cloud_processor
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.motion = self.get_motion_model()
        self.obstacles = set(zip(ox, oy))  # Store obstacles as a set for easy access
        # self.calc_obstacle_map(ox, oy)

        # Declare input and output ports
        self._grid_map_input_index = self.DeclareAbstractInputPort("grid_map", AbstractValue.Make(np.zeros((1, 1)))).get_index()
        self._next_goal_input_index = self.DeclareVectorInputPort("goal", 2).get_index()

        self._next_position_output_index = self.DeclareVectorOutputPort("next_position", 2, self.CalcNextPosition).get_index()

    def connect_processor(self, station: Diagram, builder: DiagramBuilder):
        point_cloud_processor_output = self.point_cloud_processor.GetOutputPort("grid_map")
        grid_map_input = self.get_input_port(self._grid_map_input_index)

        builder.Connect(point_cloud_processor_output, grid_map_input)

    def CalcNextPosition(self, context: Context, output: AbstractValue):
        grid_map = self.EvalAbstractInput(context, self._grid_map_input_index).get_value()
        goal = self.EvalVectorInput(context, self._next_goal_input_index).get_value()
        rx, ry = self.planning(self.current_position, goal)
        next_position = (rx[1], ry[1])
        output.set_value(next_position)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # grid index
            self.y = y  # grid index
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, start, goal):
        """
        Perform A* path planning.

        start: tuple of start position (x, y)
        goal: tuple of goal position (x, y)
        """
        start_node = self.Node(self.calc_xy_index(start[0], self.min_x),
                               self.calc_xy_index(start[1], self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(goal[0], self.min_x),
                              self.calc_xy_index(goal[1], self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while open_set:
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            # Animation
            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            # Expand search based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0
        return w * math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)
        if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
            return False
        return not self.obstacle_map[node.x][node.y]

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # Generate obstacle map
        self.obstacle_map = [[False for _ in range(self.y_width)] for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]

    def update_with_new_area(self, discovered_cells):
        """
        Update the grid dynamically with new discovered cells.
        """
        for (x, y), state in discovered_cells.items():
            if state == 1:  # Add obstacle
                self.obstacles.add((x, y))
            elif state == 0 and (x, y) in self.obstacles:  # Remove obstacle if marked free
                self.obstacles.remove((x, y))
        self.calc_obstacle_map([x for x, y in self.obstacles], [y for x, y in self.obstacles])

    def update_current_position(self, new_position):
        """
        Update the robot's current position.
        """
        self.current_position = new_position
        print(f"Updated current position to {self.current_position}")

    def set_new_goal(self, new_goal):
        """
        Set a new goal for the robot.
        """
        self.goal = new_goal
        print(f"New goal set to {self.goal}")

    def print_grid(self):
        """
        Print a subset of the grid around the start and goal for visualization.
        """
        min_x = min(self.min_x, self.current_position[0], self.goal[0]) - 1
        max_x = max(self.max_x, self.current_position[0], self.goal[0]) + 1
        min_y = min(self.min_y, self.current_position[1], self.goal[1]) - 1
        max_y = max(self.max_y, self.current_position[1], self.goal[1]) + 1

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x, y) in self.obstacles:
                    print("#", end=" ")
                elif (x, y) == self.current_position:
                    print("R", end=" ")
                elif (x, y) == self.goal:
                    print("G", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

def main():
    print("A* pathfinding with animation")

    sx = 10.0  # start x position
    sy = 10.0  # start y position
    gx = 50.0  # goal x position
    gy = 50.0  # goal y position
    grid_size = 2.0
    robot_radius = 1.0

    # Define initial obstacle positions
    ox, oy = [], []
    for i in range(-10, 60):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(60.0)
        oy.append(i)
    for i in range(-10, 61):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0)
        oy.append(i)
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    planner = DynamicPathPlanner(ox, oy, grid_size, robot_radius)

    # Update new obstacles dynamically
    new_area = {(30, 30): 1, (31, 31): 1, (32, 32): 1}
    planner.update_with_new_area(new_area)

    # Planning path from start to goal
    rx, ry = planner.planning((sx, sy), (gx, gy))

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()
