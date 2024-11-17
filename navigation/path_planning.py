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
from pydrake.trajectories import PiecewisePolynomial
from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    LeafSystem,
    Context,
    State,
    BasicVector,
    Value,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Sphere,
    RigidTransform,
    Diagram,
    Rgba,
)
from navigation.map_helpers import PointCloudProcessor

show_animation = True

class DynamicPathPlanner(LeafSystem):
    def __init__(self, station, builder, point_cloud_processor, initial_position, resolution, robot_radius, time_step=0.1, meshcat=None):
        """
        Initialize grid map for A* planning and smooth trajectory generation.

        point_cloud_processor: The PointCloudProcessor instance for obtaining the updated grid map.
        initial_position: Initial position of the robot.
        resolution: Grid resolution.
        robot_radius: Robot radius.
        time_step: Time step for trajectory updates (default: 0.1 seconds).
        meshcat: Instance of Meshcat for visualization.
        """
        LeafSystem.__init__(self)
        self.point_cloud_processor = point_cloud_processor
        self._initial_position = initial_position
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.time_step = time_step  # Store the configurable time step
        self.meshcat = meshcat
        self.grid_size = (100, 100)  # Default grid size

        # State storage
        self._current_position = self.DeclareDiscreteState(3)  # x, y, theta
        self._trajectory_flag = self.DeclareDiscreteState(1)  # Boolean flag for trajectory update
        self._done_astar = self.DeclareDiscreteState(1)  # Boolean flag for trajectory update
        self._desired_state = self.DeclareDiscreteState(10)  # Vector of size 20
        self.DeclareStateOutputPort("done_astar", self._done_astar)
        # Declare output port for desired state (position and velocity)
        self.DeclareVectorOutputPort(
            "desired_state",
            10,
            self.UpdateDesiredState
        )

        # Declare input ports
        self._grid_map_input_index = self.DeclareAbstractInputPort(
            "grid_map", AbstractValue.Make(np.zeros((100, 100)))
        ).get_index()
        self._goal_input_index = self.DeclareVectorInputPort("goal", 3).get_index()
        self._robot_position_input_index = self.DeclareVectorInputPort("robot_position", 3).get_index()
        self._execute_path_input_index = self.DeclareVectorInputPort("execute_path", 1).get_index()  # 0 or 1


        # Trajectory storage
        self.trajectory = None
        self.trajectory_time = self.DeclareDiscreteState(1)
        self.waypoints = None
        self.current_waypoint_idx = 0


        # Visualization elements
        self.visualize_path = []
        if self.meshcat:
            self.visualize_path = "planned_path"
            self.visualize_position = "current_position"

        # Periodic update for trajectory generation
        self._state_update_event = self.DeclarePeriodicUnrestrictedUpdateEvent(self.time_step, 0.0, self.UpdateTrajectory)

        position = np.array([0.0, 0.0, 0.0])
        self.arm_state = [0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0] # default stowed arm state

        # Update the discrete state
        self.desired_state = np.concatenate([position, self.arm_state])


    def connect_processor(self, station: Diagram, builder: DiagramBuilder):
        point_cloud_processor_output = self.point_cloud_processor.get_output_port(0)  # Output grid_map from point cloud processor
        grid_map_input = self.get_input_port(self._grid_map_input_index)

        builder.Connect(point_cloud_processor_output, grid_map_input)

    def UpdateDesiredState(self, context: Context, output):
        output.SetFromVector(self.desired_state)

    def UpdateTrajectory(self, context: Context, state):
        """Generate or update the trajectory based on the FSM flag and goal."""
        execute_path = bool(self.get_input_port(self._execute_path_input_index).Eval(context)[0])
        current_position = self.EvalVectorInput(context, self._robot_position_input_index).get_value()

        # Default to staying still
        position = current_position[:3]
        state.get_mutable_discrete_state(self._done_astar).set_value([0])

        if execute_path:
            goal = self.EvalVectorInput(context, self._goal_input_index).get_value()

            # Check if we've reached the goal
            distance_to_goal = np.linalg.norm(goal[:2] - current_position[:2])
            if distance_to_goal < 0.1:  # 10cm threshold
                # We've reached the goal
                state.get_mutable_discrete_state(self._done_astar).set_value([1])

            elif self.waypoints is None:
                # Calculate new A* path
                grid_map = self.EvalAbstractInput(context, self._grid_map_input_index).get_value()
                self.grid_size = grid_map.shape
                self.grid_center_index = self.grid_size[0] // 2

                # Use A* to find the path
                rx, ry = self.planning(current_position, goal, grid_map)

                if len(rx) > 1:
                    # Downsample the path to create larger segments
                    # Take every Nth point to achieve roughly 1m segments
                    desired_segment_length = 0.5  # 1 meter segments
                    points = np.vstack((rx[::-1], ry[::-1])).T  # Reverse path and combine x,y

                    # Calculate cumulative distances
                    diffs = np.diff(points, axis=0)
                    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
                    cumulative_dist = np.cumsum(segment_lengths)

                    # Select points that are approximately desired_segment_length apart
                    total_dist = cumulative_dist[-1]
                    num_segments = max(2, int(total_dist / desired_segment_length))
                    desired_distances = np.linspace(0, total_dist, num_segments)

                    # Find indices of points closest to desired distances
                    indices = [0]  # Always include start point
                    for dist in desired_distances[1:-1]:
                        idx = np.argmin(np.abs(cumulative_dist - dist))
                        indices.append(idx + 1)  # +1 because cumulative_dist is one shorter than points
                    indices.append(len(points) - 1)  # Always include end point

                    # Create waypoints with heading angles
                    selected_points = points[indices]
                    headings = np.zeros(len(selected_points))

                    # Calculate heading angles to face next waypoint
                    for i in range(len(selected_points) - 1):
                        dx = selected_points[i+1][0] - selected_points[i][0]
                        dy = selected_points[i+1][1] - selected_points[i][1]
                        headings[i] = np.arctan2(dy, dx)
                    # Final heading should match the goal's heading or maintain last segment's heading
                    headings[-1] = goal[2] if len(goal) > 2 else headings[-2]

                    # Calculate times based on 1 m/s desired speed
                    distances = np.sqrt(np.sum(np.diff(selected_points, axis=0)**2, axis=1))
                    times = np.concatenate(([0], np.cumsum(distances)))  # Time = distance when speed = 1 m/s

                    # Combine positions and headings into waypoints
                    self.waypoints = np.column_stack((selected_points, headings))
                    self.current_waypoint_idx = 0
                    position = self.waypoints[self.current_waypoint_idx]
                else:
                    print("No path found")
                    state.get_mutable_discrete_state(self._done_astar).set_value([1])

            elif self.waypoints is not None:
                # Get current waypoint
                current_waypoint = self.waypoints[self.current_waypoint_idx]

                # Check if we've reached the current waypoint
                distance_to_waypoint = np.linalg.norm(current_waypoint[:2] - current_position[:2])
                if distance_to_waypoint < 0.2:  # 10cm threshold
                    self.current_waypoint_idx += 1
                    if self.current_waypoint_idx >= len(self.waypoints):
                        state.get_mutable_discrete_state(self._done_astar).set_value([1])
                        self.waypoints = None
                        self.current_waypoint_idx = 0
                    else:
                        current_waypoint = self.waypoints[self.current_waypoint_idx]


                # Follow existing trajectory
                position = current_waypoint

        else:
            # Reset trajectory when not executing
            self.waypoints = None
            self.current_waypoint_idx = 0

        # Update the desired state with positions and default arm state
        self.desired_state= np.concatenate([position, self.arm_state])
        state.get_mutable_discrete_state(self._desired_state).SetFromVector(self.desired_state)

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
            for i, _ in enumerate(self.get_motion_model()):
                node = self.Node(current.x + self.get_motion_model()[i][0],
                                 current.y + self.get_motion_model()[i][1],
                                 current.cost + self.get_motion_model()[i][2], c_id)
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

    def calc_grid_position(self, index, resolution, grid_center_index=None):
        if grid_center_index is None:
            grid_center_index = self.grid_center_index
        return (index - grid_center_index) * resolution

    def calc_xy_index(self, position, center_index):
        return int(round((position / self.resolution) + center_index))

    def calc_grid_index(self, node):
        return node.y * self.grid_size[1] + node.x

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
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # grid index
            self.y = y  # grid index
            self.cost = cost
            self.parent_index = parent_index

    def DoCalcUnrestrictedUpdate(self, context, state):
        """Increment trajectory time during updates."""
        if self.trajectory:
            self.trajectory_time += self.time_step  # Use the configurable time step

    def visualize_path_on_grid(self, grid_map, start, goal, path_x, path_y, resolution):
        """
        Visualize the A* path on the grid map.

        Parameters:
        grid_map (ndarray): The grid map of the world (2D numpy array).
                            Values should be:
                            - 0 for free space
                            - 1 or greater for obstacles
        start (tuple): The start position as (x, y) in real-world coordinates.
        goal (tuple): The goal position as (x, y) in real-world coordinates.
        path_x (list): List of x coordinates of the planned path in real-world coordinates.
        path_y (list): List of y coordinates of the planned path in real-world coordinates.
        resolution (float): The resolution of the grid map (distance per grid cell).
        """
        # Convert start, goal, and path points from (x, y) to grid indices (ix, iy)
        def xy_to_grid(x, y, resolution, grid_map_shape):
            ix = int(round(x / resolution)) + (grid_map_shape[0] // 2)  # Offset to center grid
            iy = int(round(y / resolution)) + (grid_map_shape[1] // 2)  # Offset to center grid
            return ix, iy

        # Convert start and goal positions to grid indices
        start_ix, start_iy = xy_to_grid(start[0], start[1], resolution, grid_map.shape)
        goal_ix, goal_iy = xy_to_grid(goal[0], goal[1], resolution, grid_map.shape)

        # Convert path points to grid indices
        path_indices = [xy_to_grid(x, y, resolution, grid_map.shape) for x, y in zip(path_x, path_y)]
        path_ix, path_iy = zip(*path_indices)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Show the grid map
        ax.imshow(grid_map, cmap='gray', origin='upper')

        # Mark the start point
        ax.plot(start_iy, start_ix, "ro", label="Start", markersize=10)  # Red circle for start (note iy, ix for plotting)

        # Mark the goal point
        ax.plot(goal_iy, goal_ix, "go", label="Goal", markersize=10)  # Green circle for goal (note iy, ix for plotting)

        # Plot the planned path
        ax.plot(path_iy, path_ix, "b-", linewidth=2, label="Planned Path")  # Blue line for path (note iy, ix for plotting)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("A* Path Planning Visualization")
        ax.legend()

        # Display the plot
        plt.grid(True)
        plt.show()