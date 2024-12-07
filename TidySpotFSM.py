import random
import numpy as np
from pydrake.all import (
    LeafSystem,
    AbstractValue,
    Context,
    State,
)
from enum import Enum, auto

from navigation.Navigator import NavigationState

# Define FSM States
class SpotState(Enum):
    IDLE = auto() #IDLE is for waiting for the simulation to settle
    EXPLORE = auto()
    APPROACH_OBJECT = auto()
    GRASP_OBJECT = auto()
    TRANSPORT_OBJECT = auto()
    DEPOSIT_OBJECT = auto()
    RETURN_TO_IDLE = auto()

# Define the high-level planner for Spot
class TidySpotFSM(LeafSystem):
    def __init__(self, plant, bin_location=[0, 0, 0]):
        LeafSystem.__init__(self)

        # Internal states
        self._state_index = self.DeclareAbstractState(AbstractValue.Make(SpotState.IDLE))
        self.navigator_state_commanded = self.DeclareDiscreteState(1)

        self._times_index = self.DeclareAbstractState(AbstractValue.Make({"initial": 0.0}))
        self._attempts_index = self.DeclareDiscreteState(1)

        # Internal variables
        self.bin_location = bin_location
        self.path_planning_goal = (0, 0, 0)  # Goal for path planning, if self.path_planning_goal[2] == None then Navigator will autogenerate final heading
        
        # Input ports
        self._spot_body_state_index = self.DeclareVectorInputPort("body_poses", 20).get_index()
        self._path_planning_desired_index = self.DeclareVectorInputPort("path_planning_desired", 3).get_index()
        self._object_clusters_input = self.DeclareAbstractInputPort("detection_dict", AbstractValue.Make({}))

        # Input ports for various components
        self.DeclareVectorInputPort("object_detected", 1)
        self.DeclareVectorInputPort("navigation_complete", 1)
        self.DeclareVectorInputPort("frontier", 2)
        self.DeclareVectorInputPort("grasp_complete", 1) # TODO: Add success/fail flag to data sent to this port

        # Output ports
        # Declare output ports for the path planner. The path planner then sends to the actual robot
        self.DeclareStateOutputPort("fsm_state", self._state_index)
        self.DeclareStateOutputPort("navigator_state_commanded", self.navigator_state_commanded)
        self._path_planning_goal_output = self.DeclareVectorOutputPort("path_planning_goal", 3, self.SetPathPlanningGoal).get_index()
        self._path_planning_position_output = self.DeclareVectorOutputPort("path_planning_position", 3, self.SetPathPlanningCurrentPosition).get_index()

        # Output ports for various components
        self.do_arm_controller_mission = self.DeclareDiscreteState(1) # Use a state of 1 to represent a boolean
        self.DeclareStateOutputPort("do_arm_controller_mission", self.do_arm_controller_mission)

        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

    def connect_components(self, builder, object_detector, spot_arm_ik_controller, point_cloud_mapper, navigator, station):
        builder.Connect(station.GetOutputPort("spot.state_estimated"), self.get_input_port(self._spot_body_state_index))

        # Connect the FSM to ObjectDetector
        builder.Connect(self.GetOutputPort("fsm_state"), object_detector.GetInputPort("fsm_state"))

        # Connect the Navigator to the FSM
        builder.Connect(self.get_output_port(self._path_planning_goal_output), navigator.GetInputPort("goal"))
        builder.Connect(self.get_output_port(self._path_planning_position_output), navigator.GetInputPort("current_position"))
        builder.Connect(navigator.GetOutputPort("navigation_complete"), self.GetInputPort("navigation_complete"))
        builder.Connect(self.GetOutputPort("navigator_state_commanded"), navigator.GetInputPort("navigator_state"))

        # Connect the Mapper's detected objects dictionary to the FSM
        builder.Connect(point_cloud_mapper.GetOutputPort("object_clusters"), self._object_clusters_input) # TODO: Check that output name is correct
        # connect the mappers frontier to the FSM
        builder.Connect(point_cloud_mapper.GetOutputPort("frontier"), self.GetInputPort("frontier"))

        # Connect the grasper to the FSM
        builder.Connect(self.GetOutputPort("do_arm_controller_mission"), spot_arm_ik_controller.GetInputPort("do_arm_controller_mission"))
        builder.Connect(spot_arm_ik_controller.GetOutputPort("done_grasp"), self.GetInputPort("grasp_complete"))

    def get_spot_state_input_port(self):
        return self.GetInputPort("body_poses")

    def _initialize_state(self, context: Context, state: State):
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        self.object_clusters = self._object_clusters_input.Eval(context)

    def _get_navigation_completed(self, context, state):
        navigator_state_commanded = state.get_mutable_discrete_state(self.navigator_state_commanded).get_value()[0]
        navigation_complete = self.GetInputPort("navigation_complete").Eval(context)[0]
        return navigation_complete and navigator_state_commanded

    def _get_grasping_completed(self, context, state):
        do_arm_controller_mission = state.get_mutable_discrete_state(self.do_arm_controller_mission).get_value()[0]
        grasp_complete = self.GetInputPort("grasp_complete").Eval(context)[0]
        return grasp_complete and do_arm_controller_mission

    def Update(self, context, state):
        current_state = context.get_abstract_state(int(self._state_index)).get_value()
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        self.object_clusters = self._object_clusters_input.Eval(context)
        current_time = context.get_time()
        if current_state == SpotState.IDLE:
            state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.STOP.value])

            # select a new area to explore and go to it
            if current_time == 0.0:
                # At time 0, we explore our start location to get a good map
                self.set_new_exploration_goal((0, 0, None))
            else:
                self.set_new_random_exploration_goal(context)

            print("State: IDLE -> EXPLORE")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.EXPLORE)

        elif current_state == SpotState.EXPLORE: # TODO: Handle case where object is detected during exploration
            # Explore until an object is detected
            if self._get_navigation_completed(context, state):
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.STOP.value])
                print("ASTAR DONE RECEIVED: Exploration completed to area.")

                if self.check_detections():
                    grid_points, centroid = self.object_clusters[1].values() # if object_clusters exists, there will always be one with key value 1
                    print(f"Found object at {centroid['world']}, approaching ...")

                    self.current_object_location = centroid["world"]

                    self.approach_object(self.current_object_location)

                    print("State: EXPLORE -> APPROACH_OBJECT")
                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.APPROACH_OBJECT)
                else:
                    print("Failed to detect any objects. Transitioning to IDLE to generate new area to explore.")
                    print("State: EXPLORE -> IDLE")
                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.IDLE)
            else:
                # print(f"Exploring area at {self.path_planning_goal}")
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.GOTO_EXACT_LOCATION.value])

        elif current_state == SpotState.APPROACH_OBJECT:
            if self._get_navigation_completed(context, state):
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.STOP.value])
                print("Arrived at grasp location, ready to pick object at ", self.current_object_location)

                print("State: APPROACH_OBJECT -> GRASP_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.GRASP_OBJECT)
                # Send the grasp request to the grasper
                state.get_mutable_discrete_state(self.do_arm_controller_mission).set_value([1])
                print("Grasp requested.")
            else:
                # print("Approaching object at ", self.current_object_location)
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.MOVE_NEAR_OBJECT.value])


        elif current_state == SpotState.GRASP_OBJECT:
            if self._get_grasping_completed(context, state):
                print("Grasping object successful.")

                print(f"Transporting object to bin at {self.bin_location} ...")
                self.transport_object()

                print("State:  GRASP_OBJECT -> TRANSPORT_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.TRANSPORT_OBJECT)
            else:
                # print("Currently grasping object")
                # Assume there is no failure state here
                pass

        elif current_state == SpotState.TRANSPORT_OBJECT:
            if self._get_navigation_completed(context, state):
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.STOP.value])
                print("Arrived at bin location, ready to drop object at ", self.bin_location)

                print("State: TRANSPORT_OBJECT -> DEPOSIT_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.DEPOSIT_OBJECT)

            else:
                # print("Approaching bin at ", self.bin_location)
                state.get_mutable_discrete_state(self.navigator_state_commanded).set_value([NavigationState.MOVE_NEAR_OBJECT.value])

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

    def SetPathPlanningGoal(self, context, output):
        output.SetFromVector(self.path_planning_goal)

    def SetPathPlanningCurrentPosition(self, context, output):
        spot_body_pos = self.robot_state[:3]
        output.SetFromVector(spot_body_pos)

    def check_detections(self):
        # print("self.object_clusters:", self.object_clusters)
        return bool(self.object_clusters)

    def approach_object(self, object_location):
        self.path_planning_goal = (object_location[0], object_location[1], None)


    def transport_object(self):
        self.path_planning_goal = (self.bin_location[0], self.bin_location[1], None)

    def deposit_object(self):
        # Deposit the object into the bin
        print("Depositing object into bin...")
        # Actual deposit code here
        pass

    def set_new_random_exploration_goal(self, context):
        # Random search
        new_goal = self.GetInputPort("frontier").Eval(context)
        if new_goal is None:
            new_goal = (random.uniform(-4.0, 4.0), random.uniform(-4.0, 4.0))

        # clip the goal to slightly inside the bounds
        new_goal = (np.clip(new_goal[0], -4.5, 4.5), np.clip(new_goal[1], -4.5, 4.5))



        print(f"Exploring environment, new exploration goal: {new_goal}")

        self.path_planning_goal = (new_goal[0], new_goal[1], None)

    def set_new_exploration_goal(self, goal):
        print("Setting new exploration goal:", goal)
        self.path_planning_goal = goal