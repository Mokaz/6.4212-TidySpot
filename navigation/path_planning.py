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
    State,
    ImageRgba8U,
    ImageDepth32F,
    ImageLabel16I,
    Diagram,
    PointCloud,
)
from navigation.map_helpers import PointCloudProcessor

show_animation = True

class DynamicPathPlanner(LeafSystem):
    def __init__(self, station: Diagram, point_cloud_processor: PointCloudProcessor, initial_position, resolution, robot_radius):
        """
        Initialize grid map for A* planning.

        point_cloud_processor: The PointCloudProcessor instance for obtaining the updated grid map.
        initial_position: Initial position of the robot.
        resolution: Grid resolution.
        robot_radius: Robot radius.
        """
        LeafSystem.__init__(self)
        self.point_cloud_processor = point_cloud_processor
        self._initial_position = initial_position
        self.resolution = resolution
        self.robot_radius = robot_radius

        # Drake internal states and output port
        self._base_position = self.DeclareDiscreteState(3)
        self._done_astar = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("base_position", self._base_position)
        self.DeclareStateOutputPort("done_astar", self._done_astar)

        self.motion = self.get_motion_model()

        # Declare input and output ports
        self._grid_map_input_index = self.DeclareAbstractInputPort("grid_map", AbstractValue.Make(np.zeros((100, 100)))).get_index()
        self._next_goal_input_index = self.DeclareVectorInputPort("goal", 3).get_index()
        self._robot_position_input_index = self.DeclareVectorInputPort("robot_position", 3).get_index()

        self._next_position_output_index = self.DeclareVectorOutputPort("next_position", 3, self.CalcNextPosition).get_index()

        # Initialize and update
        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(period_sec=0.1, offset_sec=0.0, update=self.calc_next_position)

    def connect_processor(self, station: Diagram, builder: DiagramBuilder):
        point_cloud_processor_output = self.point_cloud_processor.get_output_port(0)  # Output grid_map from point cloud processor
        grid_map_input = self.get_input_port(self._grid_map_input_index)

        builder.Connect(point_cloud_processor_output, grid_map_input)

    def CalcNextPosition(self, context: Context, output: AbstractValue):
        output.set_value(self.next_position)

    def _initialize_state(self, context: Context, state):
        state.get_mutable_discrete_state(self._base_position).set_value(self._initial_position)
        state.get_mutable_discrete_state(self._done_astar).set_value([0])

    def get_spot_state_input_port(self):
        return self.get_input_port(self._robot_position_input_index)

    def _get_current_position(self, context: Context):
        return self.get_spot_state_input_port().Eval(context)

    def calc_next_position(self, context: Context, state):
        grid_map = self.EvalAbstractInput(context, self._grid_map_input_index).get_value()
        goal = self.EvalVectorInput(context, self._next_goal_input_index).get_value()
        self.current_position = self._get_current_position(context)
        rx, ry = self.planning(self.current_position, goal, grid_map)
        if len(rx) > 1:
            self.next_position = (rx[1], ry[1])
        else:
            self.next_position = self.current_position[:2]

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # grid index
            self.y = y  # grid index
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, start, goal, grid_map):
        """
        Perform A* path planning using the provided grid_map.

        start: tuple of start position (x, y)
        goal: tuple of goal position (x, y)
        grid_map: The updated grid map from PointCloudProcessor.
        """
        start_node = self.Node(self.calc_xy_index(start[0], grid_map.shape[1] // 2),
                               self.calc_xy_index(start[1], grid_map.shape[0] // 2), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(goal[0], grid_map.shape[1] // 2),
                              self.calc_xy_index(goal[1], grid_map.shape[0] // 2), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while open_set:
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

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

                if not self.verify_node(node, grid_map):
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
        rx, ry = [self.calc_grid_position(goal_node.x, self.resolution)], [self.calc_grid_position(goal_node.y, self.resolution)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.resolution))
            ry.append(self.calc_grid_position(n.y, self.resolution))
            parent_index = n.parent_index
        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0
        return w * math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, resolution):
        return index * resolution

    def calc_xy_index(self, position, center_index):
        return round((position + (center_index * self.resolution)) / self.resolution)

    def calc_grid_index(self, node):
        return node.y * 100 + node.x  # Assuming grid_map of size 100x100

    def verify_node(self, node, grid_map):
        """
        Verifies if the node is traversable.
        The robot can only travel in free space (0) or unexplored space (-1).
        """

        if node.x < 0 or node.y < 0 or node.x >= grid_map.shape[1] or node.y >= grid_map.shape[0]:
            return False
        return grid_map[node.x, node.y] == 0 or grid_map[node.x, node.y] == -1

    @staticmethod
    def get_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [-1, -1, math.sqrt(2)], [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)], [1, 1, math.sqrt(2)]]
