import random
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    AbstractValue,
    Context,
    State,
    PiecewisePose,
    PiecewisePolynomial,
    RandomGenerator,
    RigidTransform,
    RollPitchYaw,
    Parser,
    AddMultibodyPlantSceneGraph,
    UniformlyRandomRotationMatrix,
    Simulator,
    AbstractStateIndex,
)
from enum import Enum, auto
from navigation.FrontierExplorer import FrontierExplorer
from navigation.PointCloudMapper import PointCloudMapper
from navigation.DynamicPathPlanner import DynamicPathPlanner


# Define FSM States
class SpotState(Enum):
    IDLE = auto() #IDLE is for waiting for the simulation to settle
    EXPLORE = auto()
    APPROACH_OBJECT = auto()
    GRASP_OBJECT = auto()
    TRANSPORT_OBJECT = auto()
    DEPOSIT_OBJECT = auto()
    RETURN_TO_IDLE = auto()

class NavigationGoalType(Enum):
    EXPLORE = auto()
    APPROACH_OBJECT = auto()
    DEPOSIT_OBJECT = auto()

# Define the high-level planner for Spot
class TidySpotPlanner(LeafSystem):
    def __init__(self, plant):
        LeafSystem.__init__(self)

        # Spot's internal planning states
        self._state_index = self.DeclareAbstractState(AbstractValue.Make(SpotState.IDLE))

        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)
        self._exploring = self.DeclareDiscreteState(1)
        self.next_arm_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.next_desired_position = np.array([0.0, 0.0, 0.0])
        # Input ports
        self._spot_body_state_index = self.DeclareVectorInputPort("body_poses", 20).get_index()
        self._path_planning_desired_index = self.DeclareVectorInputPort("path_planning_desired", 3).get_index()
        self._detection_dict_input = self.DeclareAbstractInputPort("detection_dict", AbstractValue.Make({}))

        # Input ports for various components
        self.DeclareVectorInputPort("object_detected", 1)
        self.DeclareVectorInputPort("path_planning_finished", 1)

        self.DeclareVectorInputPort("grasp_completed", 1) # TODO: Add success/fail flag to data sent to this port

        # Output ports
        # Declare output ports for the path planner. The path planner then sends to the actual robot
        self.use_path_planner = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("use_path_planner", self.use_path_planner)
        self._path_planning_goal_output = self.DeclareVectorOutputPort("path_planning_goal", 3, self.CalcPathPlanningGoal).get_index()
        self._path_planning_position_output = self.DeclareVectorOutputPort("path_planning_position", 3, self.CalcPathPlanningPosition).get_index()

        # Output ports for various components
        self.DeclareVectorOutputPort("request_grasp", 1, lambda context, output: print("Setting request_grasp output...")) # TODO: Replace lambda with function that sets output based on criteria (attempth count etc.)

        self._rng = np.random.default_rng()


        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)


        # Declare internal variables
        self.detected_objects = []
        self.explored_area = set()  # Grid to track explored areas
        self.bin_location = (0, 0)  # Assume bin is at a fixed location
        self.path_planning_goal = (0, 0, 0)  # Goal for path planning


    def connect_components(self, builder, grasper, point_cloud_mapper, dynamic_path_planner, station):
        builder.Connect(station.GetOutputPort("spot.state_estimated"), self.get_input_port(self._spot_body_state_index))
        # builder.Connect(self.get_output_port(self._spot_commanded_state_index), station.GetInputPort("spot.desired_state"))


        # Connect the path planner to the FSM planner
        # The path planner uses the robot's current position and the goal to plan a path, and returns a flag when it's done. It only moves if use_path_planner is set to 1.
        builder.Connect(self.get_output_port(self._path_planning_goal_output), dynamic_path_planner.GetInputPort("goal"))
        builder.Connect(self.get_output_port(self._path_planning_position_output), dynamic_path_planner.GetInputPort("robot_position"))
        builder.Connect(dynamic_path_planner.GetOutputPort("done_astar"), self.GetInputPort("path_planning_finished"))
        builder.Connect(self.GetOutputPort("use_path_planner"), dynamic_path_planner.GetInputPort("execute_path"))

        # Connect the detection dict to the FSM planner
        builder.Connect(point_cloud_mapper.GetOutputPort("detection_dict"), self._detection_dict_input) # TODO: Check that output name is correct

        # Connect the grasper to the FSM planner
        builder.Connect(self.GetOutputPort("request_grasp"), grasper.GetInputPort("do_grasp"))
        builder.Connect(grasper.GetOutputPort("done_grasp"), self.GetInputPort("grasp_completed"))

    def get_spot_state_input_port(self):
        return self.GetInputPort("body_poses")

    def _initialize_state(self, context: Context, state: State):
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        self.detection_dict = self._detection_dict_input.Eval(context)
        state.get_mutable_discrete_state(self._exploring).set_value([0]) # TODO Unused?

    def _get_navigation_completed(self, context, state):
        # print(self.GetInputPort("path_planning_finished").Eval(context)[0])
        return self.GetInputPort("path_planning_finished").Eval(context)[0]

    def Update(self, context, state):
        current_state = context.get_abstract_state(int(self._state_index)).get_value()
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        self.detection_dict = self._detection_dict_input.Eval(context)
        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()


        if current_state == SpotState.IDLE:
            # Start exploring the environment
            print("State: IDLE -> EXPLORE")

            state.get_mutable_discrete_state(self.use_path_planner).set_value([0])
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.EXPLORE)

            # select a new area to explore and go to it
            self.set_new_random_exploration_goal()

        elif current_state == SpotState.EXPLORE:
            # Explore until an object is detected
            if self._get_navigation_completed(context, state):
                print("Exploration completed to area.")
                if self.check_detections():
                    object_index, object_location = list(self.detection_dict.items())[0] # TODO: Make this a function that chooses the best object to approach
                    print(f"Found {object_index} at {object_location}")

                    self.current_object_location = object_location

                    print("State: EXPLORE -> APPROACH_OBJECT")
                    self.approach_object(self.current_object_location)

                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.APPROACH_OBJECT)
                else:
                    print("Failed to detect any objects. Calculate new goal for EXPLORE.")
                    self.set_new_random_exploration_goal()
                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.IDLE)
            else:
                # print(f"Exploring area at {self.path_planning_goal}")
                # state.get_mutable_discrete_state(self.use_path_planner).set_value([NavigationGoalType.EXPLORE.value])
                state.get_mutable_discrete_state(self.use_path_planner).set_value([NavigationGoalType.APPROACH_OBJECT.value])

        elif current_state == SpotState.APPROACH_OBJECT:

            if self._get_navigation_completed(context, state):
                print("Arrived at object location", self.current_object_location)


                # print("State: APPROACH_OBJECT -> GRASP_OBJECT")
                # state.get_mutable_abstract_state(
                #     int(self._state_index)
                # ).set_value(SpotState.GRASP_OBJECT)

            else:
                print("Approaching object at ", self.current_object_location)
                state.get_mutable_discrete_state(self.use_path_planner).set_value([NavigationGoalType.APPROACH_OBJECT.value])


        elif current_state == SpotState.GRASP_OBJECT:
            if self.grasp_object():
                print("State:  GRASP_OBJECT -> TRANSPORT_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.TRANSPORT_OBJECT)
            else:
                print("Failed to grasp object. Returning to EXPLORE.")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.EXPLORE)

        elif current_state == SpotState.TRANSPORT_OBJECT:
            self.transport_object()
            print("State: TRANSPORT_OBJECT -> DEPOSIT_OBJECT")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.DEPOSIT_OBJECT)

        elif current_state == SpotState.DEPOSIT_OBJECT:
            self.deposit_object()
            print("State: DEPOSIT_OBJECT -> RETURN_TO_IDLE")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.RETURN_TO_IDLE)

        elif current_state == SpotState.RETURN_TO_IDLE:
            print("Returning to IDLE...")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.IDLE)

    def CalcPathPlanningGoal(self, context, output):
        output.SetFromVector(self.path_planning_goal)

    def CalcPathPlanningPosition(self, context, output):
        spot_body_pos = self.robot_state[:3]
        output.SetFromVector(spot_body_pos)

    def detect_object(self): # TODO: Remove this function
        # We check the segmentation to see if we got any objects
        objects = []
        self.detected_objects.extend(objects)
        if len(self.detected_objects) > 0:
            return True
        return False
    
    def check_detections(self):
        return bool(self.detection_dict)

    def approach_object(self, object_location):
        print(f"Navigating to object at {object_location}")

        self.path_planning_goal


        # Actual navigation code here
        pass

    def grasp_object(self):
        # Grasp the object using AnyGrasp or other methods
        print("Grasping the object...")
        # Actual grasping code here
        success = True  # Assume success in simulation
        return success

    def transport_object(self):
        # Move towards the bin to deposit the object
        print("Transporting object to bin...")
        # Actual transport code here
        pass

    def deposit_object(self):
        # Deposit the object into the bin
        print("Depositing object into bin...")
        # Actual deposit code here
        pass

    def set_new_random_exploration_goal(self):
        # Random search
        new_goal = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        self.explored_area.add(new_goal)

        print(f"Exploring environment, new exploration goal: {new_goal}")

        self.path_planning_goal = (new_goal[0], new_goal[1], 0.0)
