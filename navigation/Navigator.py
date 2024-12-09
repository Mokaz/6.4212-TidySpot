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
from scipy.interpolate import CubicSpline
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
from enum import Enum, auto
from scipy.ndimage import binary_dilation, binary_erosion
from utils.utils import (
    add_sphere_to_meshcat_xy_plane,
    convert_to_grid_coordinates,
    convert_to_world_coordinates,
)

class NavigationState(Enum):
    STOP = 0
    GOTO_EXACT_LOCATION = auto()
    MOVE_NEAR_OBJECT = auto()

class Navigator(LeafSystem):
    def __init__(self, station, builder, initial_position, resolution, bin_location, robot_length=1.3, robot_width=0.7, time_step=0.1, meshcat=None, visualize=False):
        """
        Initialize grid map for A* planning and smooth trajectory generation.

        initial_position: Initial position of the robot.
        resolution: Grid resolution.
        time_step: Time step for trajectory updates (default: 0.1 seconds).
        meshcat: Instance of Meshcat for visualization.
        """
        LeafSystem.__init__(self)
        self._initial_position = initial_position
        self.resolution = resolution
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.time_step = time_step  # Store the configurable time step
        self.meshcat = meshcat
        self.visualize = visualize
        self.grid_size = (100, 100)  # Default grid size

        # State storage
        self._current_position = self.DeclareDiscreteState(3)  # x, y, theta
        self._trajectory_flag = self.DeclareDiscreteState(1)  # Boolean flag for trajectory update
        self._navigation_complete = self.DeclareDiscreteState(1)  # Boolean flag for trajectory update
        self._spot_commanded_position = self.DeclareDiscreteState(3)  # Vector of size 3
        self._previous_navigator_state = self.DeclareDiscreteState([0])  # Flag to ensure self.goal is set only once per mission. Initialize with 0 (False)

        # Input ports
        self._grid_map_input_index = self.DeclareAbstractInputPort("grid_map", AbstractValue.Make(np.zeros((100, 100)))).get_index()
        self._goal_input_index = self.DeclareVectorInputPort("goal", 3).get_index()
        self._current_position_input_index = self.DeclareVectorInputPort("current_position", 3).get_index()
        self._navigator_state_input_index = self.DeclareVectorInputPort("navigator_state", 1).get_index()

        # Output ports
        self.DeclareStateOutputPort("navigation_complete", self._navigation_complete)
        self.DeclareVectorOutputPort("spot_commanded_position", 3, self.UpdateDesiredState)

        # Trajectory storage
        self.trajectory = None
        self.trajectory_time = self.DeclareDiscreteState(1)
        self.waypoints = None
        self.current_waypoint_idx = 0
        self.goal = None
        self.iters_at_current_waypoint = 0
        self.max_iters_at_current_waypoint = 10
        self.bin_location = bin_location

        # Periodic update for trajectory generation
        self._state_update_event = self.DeclarePeriodicUnrestrictedUpdateEvent(self.time_step, 0.0, self.UpdateTrajectory)

        # Initialize desired position
        self.spot_commanded_position = np.array([0.0, 0.0, 0.0])
        self.downsample = False
        self.inflate_obstacles = True
        self.allow_unknown_pathing = True
        self.goal_object_location = None
        self.max_rotation_degrees = 10
        self.grid_map = np.zeros((100, 100))
        self.hijacked_goal = None

    def connect_mapper(self, point_cloud_mapper, station: Diagram, builder: DiagramBuilder):
        builder.Connect(point_cloud_mapper.get_output_port(0), self.get_input_port(self._grid_map_input_index)) # Output grid_map from mapper to input grid_map of planner

    def UpdateDesiredState(self, context: Context, output):
        output.SetFromVector(self.spot_commanded_position)

    def check_unoccupied(self, position, grid_map):
        ix, iy = convert_to_grid_coordinates(position[0], position[1], self.resolution, grid_map.shape)
        if grid_map[ix, iy] == 0 or grid_map[ix, iy] == -1:
            return position
        else:
            # print("Position is occupied. Finding nearest unoccupied position.")
            # Find nearest unoccupied position
            for i in range(1, 10):
                for j in range(-i, i+1):
                    for k in range(-i, i+1):
                        if grid_map[ix+j, iy+k] == 0:
                            return convert_to_world_coordinates(ix+j, iy+k, self.resolution, grid_map.shape)

    def UpdateTrajectory(self, context: Context, state):
        """Generate or update the trajectory based on the FSM flag and goal."""
        navigator_state = int(self.get_input_port(self._navigator_state_input_index).Eval(context)[0])
        current_position = self.EvalVectorInput(context, self._current_position_input_index).get_value()
        previous_navigator_state = context.get_discrete_state(self._previous_navigator_state).get_value()[0]

        navigation_complete = context.get_discrete_state(self._navigation_complete).get_value()[0]

        if self.goal is not None:
            # Default to use old goal as desired position with curent heading
            desired_position = (self.goal[0], self.goal[1], current_position[2])
        else:
            desired_position = current_position[:3]

        if navigator_state and not navigation_complete:

            if not previous_navigator_state:
                # Set the goal when navigator_state goes from 0 to non-zero
                # print("RISING EDGE: Setting new goal")
                self.iters_at_current_waypoint = 0
                self.goal = self.EvalVectorInput(context, self._goal_input_index).get_value().copy()
                # print("Received New goal: ({:.3f}, {:.3f}, {:.3f})".format(*self.goal))

                if self.visualize and self.meshcat:
                    add_sphere_to_meshcat_xy_plane(self.meshcat, "goal_original", self.goal, radius=0.05, rgba=[0, 0, 1, 1])

                if navigator_state == NavigationState.MOVE_NEAR_OBJECT.value:
                    if np.allclose(self.goal,self.bin_location, atol=0.01):
                        approach_distance = 1
                    else:
                        approach_distance = 0.85  # Distance to stop before the object (in meters)
                    direction_vector = self.goal[:2] - current_position[:2]
                    distance_to_goal = np.linalg.norm(direction_vector)
                    if distance_to_goal > approach_distance:
                        adjusted_goal_position = current_position[:2] + (direction_vector / distance_to_goal) * (distance_to_goal - approach_distance)
                        adjusted_goal_position = self.check_unoccupied(adjusted_goal_position, self.grid_map)
                        # calculate the heading between the adjusted goal position and the original goal position
                        # This way we ensure we rotate towards the original goal position
                        self.goal[2] = np.arctan2(self.goal[1] - adjusted_goal_position[1], self.goal[0] - adjusted_goal_position[0])
                        self.goal[:2] = adjusted_goal_position
                        # print(f"Adjusted goal position for approach: {self.goal[:2]}")
                        if self.visualize and self.meshcat:
                            add_sphere_to_meshcat_xy_plane(self.meshcat, "adjusted_goal", self.goal, radius=0.05, rgba=[0, 0, 1, 1])
                    else:
                        print("Already within approach distance of object. Ending approach.")
                        state.get_mutable_discrete_state(self._navigation_complete).set_value([1])
                        desired_position = current_position[:3]

            # if we don't have waypoints already and we haven't timed out of being at the waypoint
            if self.waypoints is not None and self.iters_at_current_waypoint < self.max_iters_at_current_waypoint:
                self.iters_at_current_waypoint += 1
                current_waypoint = self.waypoints[self.current_waypoint_idx]

                # if we are about a meter away from the last waypoint, update the map
                if self.current_waypoint_idx == len(self.waypoints) - 10:
                    self.grid_map = self.EvalAbstractInput(context, self._grid_map_input_index).get_value()

                # Check if we've reached the current waypoint
                distance_to_waypoint = np.linalg.norm(current_waypoint[:2] - current_position[:2])

                # Normalize the angular difference to the range [-pi, pi]
                angle_difference = (current_position[2] - current_waypoint[2] + np.pi) % (2 * np.pi) - np.pi
                angle_okay = abs(angle_difference) < np.radians(self.max_rotation_degrees)  # about 10 degrees

                threshold = 0.1
                # iteratively check nodes, since we might satisfy multiple conditions of sequential nodes
                while distance_to_waypoint < threshold and angle_okay:  # 10cm threshold

                    if self.current_waypoint_idx == len(self.waypoints) - 1:
                        angle_difference = (current_position[2] - current_waypoint[2] + np.pi) % (2 * np.pi) - np.pi
                        angle_okay = abs(angle_difference) < np.radians(self.max_rotation_degrees)  # about 10 degrees
                        threshold = 0.1
                        if not (distance_to_waypoint < threshold and angle_okay):
                            # if we haven't satisfied the stricter conditions of the last node, break
                            break
                        # otherwise, allow the last node to be satisfied
                    # increment the node
                    self.current_waypoint_idx += 1
                    self.iters_at_current_waypoint = 0
                    if self.current_waypoint_idx >= len(self.waypoints):
                        print("Reached final waypoint, DONE ASTAR")
                        if self.hijacked_goal is None:
                            state.get_mutable_discrete_state(self._navigation_complete).set_value([1])
                        self.waypoints = None
                        self.current_waypoint_idx = 0
                        break
                    else:
                        current_waypoint = self.waypoints[self.current_waypoint_idx]
                        distance_to_waypoint = np.linalg.norm(current_waypoint[:2] - current_position[:2])
                        angle_difference = (current_position[2] - current_waypoint[2] + np.pi) % (2 * np.pi) - np.pi
                        angle_okay = abs(angle_difference) < np.radians(self.max_rotation_degrees)  # about 10 degrees
                        # update the map

                # Follow existing trajectory
                desired_position = current_waypoint
            # also need this iters check to see if we're stuck, then replan
            elif self.iters_at_current_waypoint >= self.max_iters_at_current_waypoint:
                self.hijacked_goal = self.goal
                print("Stuck at waypoint. Replanning.")
                # use the gridmap and select nodes that are not occupied and are completely free, move there while maintaining same robot pose
                new_waypoints = self.find_nearest_free_space_with_heading(current_position, self.grid_map)
                self.waypoints = new_waypoints
                self.current_waypoint_idx = 0
                self.iters_at_current_waypoint = 0

            elif self.waypoints is None:
                self.iters_at_current_waypoint = 0
                # print("Generating new A* path to:", self.goal)
                # Calculate new A* path
                self.hijacked_goal = None
                self.grid_map = self.EvalAbstractInput(context, self._grid_map_input_index).get_value()
                self.grid_size = self.grid_map.shape
                self.grid_center_index = self.grid_size[0] // 2

                if self.inflate_obstacles:
                    if self.allow_unknown_pathing:
                        grid_map = np.where(self.grid_map == -1, 0, self.grid_map)  # Replace -1 with 0

                    # Inflate obstacles in the grid map
                    grid_map = binary_dilation(grid_map, iterations=2) # the robot is 1.1m long and 0.5m wide, if we do 2 iterations, obstacles expand to 0.4 around. hopefully this is enough

                # Use A* to find the path
                self.goal[:2] = self.check_unoccupied(self.goal[:2], self.grid_map)
                rx, ry = self.planning(current_position, self.goal, grid_map)

                if len(rx) > 1:
                    # when the segments are too small, the robot movement is not smooth
                    # Downsample the path to create larger segments
                    # Take every Nth point to achieve roughly 1m segments
                    if self.downsample:
                        desired_segment_length = 0.3  # 1 meter segments
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
                    else:
                        selected_points = np.vstack((rx[::-1], ry[::-1])).T  # Reverse path and combine x,y
                    headings = np.zeros(len(selected_points))

                    # Calculate heading angles to face next waypoint
                    for i in range(len(selected_points) - 1):
                        dx = selected_points[i+1][0] - selected_points[i][0]
                        dy = selected_points[i+1][1] - selected_points[i][1]
                        headings[i] = np.arctan2(dy, dx)
                    # Final heading should match the goal's heading or maintain last segment's heading
                    if np.isnan(self.goal[2]):
                        headings[-1] = headings[-2]  # Use the previous heading
                        self.goal[2] = headings[-2]  # Update the goal's heading
                        # print("self.goal[2] is nan. Setting final heading, new goal:", self.goal)
                    else:
                        headings[-1] = self.goal[2]  # Use the goal's heading

                    # start heading should be current robot heading
                    headings[0] = current_position[2]

                    # self.waypoints = self.generate_smooth_trajectory(selected_points, headings, resolution=0.3)
                    smoothed_headings = self.generate_smooth_headings(selected_points, headings)

                    # Ensure the final heading is achieved by adding extra points if necessary
                    while abs((smoothed_headings[-1] - headings[-1])%np.pi) > np.radians(self.max_rotation_degrees):
                        delta = smoothed_headings[-1] - headings[-1]
                        delta = (delta + np.pi) % (2 * np.pi) - np.pi
                        additional_heading = smoothed_headings[-1] - np.sign(delta) * np.radians(self.max_rotation_degrees)

                        # Add an extra point at the beginning with adjusted heading
                        selected_points = np.vstack((selected_points[0], selected_points))
                        smoothed_headings = [(angle - np.sign(delta) *  np.radians(self.max_rotation_degrees)) for angle in smoothed_headings]
                        smoothed_headings = np.insert(smoothed_headings, 0, headings[0])

                    # Combine positions and headings into waypoints
                    self.waypoints = np.column_stack((selected_points, smoothed_headings))
                    self.current_waypoint_idx = 0
                    desired_position = self.waypoints[self.current_waypoint_idx]

                    # # Snap goal to gridmap point
                    # self.goal[:2] = self.waypoints[-1][:2]

                    # print(f"New waypoints set len {len(self.waypoints)}, first desired_position:", desired_position)
                    # Visualize goal and path using meshcat
                    if self.visualize and self.meshcat:
                        # Clear existing visualization
                        self.meshcat.Delete("planned_path")
                        # Draw green lines between waypoints
                        waypoint_points = np.vstack((selected_points.T, np.zeros(len(selected_points))))
                        self.meshcat.SetLine(
                            "planned_path",
                            waypoint_points,
                            rgba=Rgba(0, 1, 0, 1),
                            line_width=4
                        )

                        # Draw red spheres at each waypoint
                        for i, point in enumerate(selected_points):
                            self.meshcat.SetObject(
                                f"planned_path/point_{i}",
                                Sphere(radius=0.01),
                                Rgba(1, 0, 0, 1)
                            )
                            self.meshcat.SetTransform(
                                f"planned_path/point_{i}",
                                RigidTransform([point[0], point[1], 0])
                            )
                else:
                    state.get_mutable_discrete_state(self._navigation_complete).set_value([1])
                    desired_position = current_position[:3]


        else:
            # Reset trajectory when not executing
            self.waypoints = None
            if not navigator_state:
                state.get_mutable_discrete_state(self._navigation_complete).set_value([0])
                self.waypoints = None
                self.current_waypoint_idx = 0

        if self.visualize and self.meshcat:
            add_sphere_to_meshcat_xy_plane(self.meshcat, "desired_position", desired_position[:2], radius=0.05, rgba=[1, 1, 0, 1])

        # print("Desired position: ({:.3f}, {:.3f}, {:.3f})".format(*desired_position))

        self.spot_commanded_position = np.array(desired_position)

        if np.any(np.isnan(self.spot_commanded_position)):
            print("Warning: spot_commanded_position contains NaN values. Resetting to current position.")
            self.spot_commanded_position = np.array(current_position[:3])

        state.get_mutable_discrete_state(self._spot_commanded_position).SetFromVector(self.spot_commanded_position)
        state.get_mutable_discrete_state(self._previous_navigator_state).set_value([navigator_state])

    def find_nearest_free_space_with_heading(self, current_position, grid_map):
        """
        Find a free space away from the nearest object while maintaining the same heading.

        Args:
            current_position (np.ndarray): Current position of the robot [x, y, theta].
            grid_map (np.ndarray): 2D array representing the grid map (0 for free, >0 for occupied).

        Returns:
            np.ndarray: Waypoints moving away from obstacles while maintaining heading.
        """
        # Apply binary erosion to ensure the robot fits in the free space
        robot_clearance = int(max(self.robot_length, self.robot_width) / self.resolution)
        eroded_grid = binary_erosion(grid_map == 0, structure=np.ones((robot_clearance, robot_clearance))).astype(int)

        # Convert the current position to grid coordinates
        cx, cy = convert_to_grid_coordinates(*current_position[:2], self.resolution, grid_map.shape)

        # Find the nearest occupied cell
        nearest_obstacle = None
        min_dist = float('inf')

        for x in range(grid_map.shape[0]):
            for y in range(grid_map.shape[1]):
                if not eroded_grid[x, y] and grid_map[x, y] > 0:  # Check if the cell is occupied
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array([x, y]))
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obstacle = np.array([x, y])

        if nearest_obstacle is not None:
            # Calculate a direction away from the obstacle
            obstacle_world = convert_to_world_coordinates(*nearest_obstacle, self.resolution, grid_map.shape)
            current_world = current_position[:2]
            direction = current_world - obstacle_world
            direction /= np.linalg.norm(direction)  # Normalize

            # Create waypoints moving away in the same heading direction
            waypoints = []
            for i in range(1, 4):  # Create 3 waypoints at 0.1m intervals
                new_position = current_world + direction * i * 0.1
                waypoints.append([new_position[0], new_position[1], current_position[2]])

            return np.array(waypoints)

        print("No nearby obstacles to avoid. Staying in place.")
        return np.array([[current_position[0], current_position[1], current_position[2]]])

    def generate_smooth_headings(self, selected_points, headings):
        """
        Smooth out the headings to ensure the change between consecutive headings is no more than 30 degrees.

        Args:
            selected_points (np.ndarray): Nx2 array of waypoints [x, y].
            headings (np.ndarray): N array of headings (in radians).

        Returns:
            np.ndarray: Smoothed headings array.
        """
        max_delta_heading = np.radians(self.max_rotation_degrees)  # Maximum allowed change in heading (30 degrees)
        smoothed_headings = headings.copy()

        for i in range(1, len(smoothed_headings)):
            delta = smoothed_headings[i] - smoothed_headings[i - 1]
            # Normalize delta to be between -pi and pi
            # delta = (delta + np.pi) % (2 * np.pi) - np.pi

            if abs(delta) > max_delta_heading:
                smoothed_headings[i] = smoothed_headings[i - 1] + np.sign(delta) * max_delta_heading

        return smoothed_headings

    def generate_smooth_trajectory(self, selected_points, headings, resolution=0.3):
        """
        Generate a smooth trajectory by interpolating waypoints and headings.

        Args:
            selected_points (np.ndarray): Nx2 array of waypoints [x, y].
            headings (np.ndarray): N array of headings (in radians).
            resolution (float): Desired segment length for smooth trajectory.

        Returns:
            np.ndarray: Smooth trajectory as an Nx3 array [x, y, heading].
        """
        # Calculate cumulative distances along waypoints
        if selected_points.shape[0] < 2:
            return np.column_stack(selected_points, headings)
        segment_lengths = np.sqrt(np.sum(np.diff(selected_points, axis=0)**2, axis=1))
        cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

        # Fit a cubic spline for headings
        heading_spline = CubicSpline(cumulative_distances, headings, bc_type='clamped')

        # Generate smooth distances for the trajectory
        total_distance = cumulative_distances[-1]
        smooth_distances = np.linspace(0, total_distance, int(total_distance / resolution))

        # Interpolate smooth positions and headings
        smooth_x = np.interp(smooth_distances, cumulative_distances, selected_points[:, 0])
        smooth_y = np.interp(smooth_distances, cumulative_distances, selected_points[:, 1])
        smooth_headings = heading_spline(smooth_distances)

        # Combine into a single trajectory array
        smooth_trajectory = np.column_stack((smooth_x, smooth_y, smooth_headings))

        return smooth_trajectory

    def planning(self, start, goal, grid_map):
        """
        Perform A* path planning using the provided grid_map.

        start: tuple of start position (x, y)
        goal: tuple of goal position (x, y)
        grid_map: The updated grid map from PointCloudMapper.
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

    def is_point_in_obstacle(self, point, grid_map):
        """
        Check if a given point lies within an obstacle or its expanded boundary in the grid_map.
        """
        x_index = self.calc_xy_index(point[0], grid_map.shape[1] // 2)
        y_index = self.calc_xy_index(point[1], grid_map.shape[0] // 2)

        if x_index < 0 or y_index < 0 or x_index >= grid_map.shape[1] or y_index >= grid_map.shape[0]:
            return True
        return grid_map[y_index, x_index] != 0

    def is_path_segment_in_obstacle(self, point1, point2, grid_map):
        """
        Check if the line segment between two points intersects with any obstacles in the grid_map.
        """
        num_samples = int(np.ceil(np.linalg.norm(point2 - point1) / self.resolution))
        for t in np.linspace(0, 1, num_samples):
            intermediate_point = (1 - t) * point1 + t * point2
            if self.is_point_in_obstacle(intermediate_point, grid_map):
                return True
        return False

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
