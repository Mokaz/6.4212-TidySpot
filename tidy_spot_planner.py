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
from navigation.exploration import FrontierExplorer
from navigation.map_helpers import PointCloudProcessor
from navigation.path_planning import DynamicPathPlanner


# Define FSM States
class SpotState(Enum):
    IDLE = auto() #IDLE is for waiting for the simulation to settle
    EXPLORE = auto()
    DETECT_OBJECT = auto()
    APPROACH_OBJECT = auto()
    GRASP_OBJECT = auto()
    TRANSPORT_OBJECT = auto()
    DEPOSIT_OBJECT = auto()
    RETURN_TO_IDLE = auto()

# Define the high-level planner for Spot
class TidySpotPlanner(LeafSystem):
    def __init__(self, plant, dynamic_path_planner):
        LeafSystem.__init__(self)

        self.dynamic_path_planner = dynamic_path_planner
        # self.controller = controller

        # Spot's internal planning states
        self._state_index = self.DeclareAbstractState(AbstractValue.Make(SpotState.IDLE))
        # self._traj_X_G_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePose()))
        # self._traj_wsg_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePolynomial()))

        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)
        self._exploring = self.DeclareDiscreteState(1)
        self.next_arm_goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.next_desired_position = np.array([0.0, 0.0, 0.0])
        # Input ports
        self._spot_body_state_index = self.DeclareVectorInputPort("body_poses", 20).get_index()
        #self._grasp_input_index = self.DeclareAbstractInputPort("grasp_selection", AbstractValue.Make((np.inf, RigidTransform()))).get_index()
        #self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()
        self._path_planning_desired_index = self.DeclareVectorInputPort("path_planning_desired", 3).get_index()

        # Input ports for various components
        self.DeclareVectorInputPort("object_detected", 1)
        self.DeclareVectorInputPort("path_planning_finished", 1)

        self.DeclareVectorInputPort("grasp_completed", 1) # TODO: Add success/fail flag to data sent to this port

        # Output ports
        # self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()), self.CalcGripperPose)
        self._spot_commanded_state_index = self.DeclareVectorOutputPort("spot_commanded_state", 20, self.CalcSpotState).get_index()

        # Declare output ports for the path planner
        self.use_path_planner = self.DeclareDiscreteState(1)
        self.DeclareStateOutputPort("use_path_planner", self.use_path_planner)
        self._target_base_position = self.DeclareVectorOutputPort("target_base_position", 3, self.CalcSpotState).get_index()
        self._path_planning_goal_output = self.DeclareVectorOutputPort("path_planning_goal", 3, self.CalcPathPlanningGoal).get_index()
        self._path_planning_position_output = self.DeclareVectorOutputPort("path_planning_position", 3, self.CalcPathPlanningPosition).get_index()
    
        # Output ports for various components
        self.DeclareVectorOutputPort("request_grasp", 1, lambda context, output: print("Setting request_grasp output...")) # TODO: Replace lambda with function that sets output based on criteria (attempth count etc.)

        # Declare output ports for the controller
        self._controller_base_pos_output = self.DeclareVectorOutputPort("desired_base_position", 3, self.SendControllerPathOutput).get_index()
        self._controller_arm_pos_output = self.DeclareVectorOutputPort("desired_arm_position", 7, self.SendControllerArmOutput).get_index()

        self._rng = np.random.default_rng()


        self.DeclareInitializationUnrestrictedUpdateEvent(self._initialize_state)
        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)


        # Declare internal variables
        self.detected_objects = []
        self.explored_area = set()  # Grid to track explored areas
        self.bin_location = (0, 0)  # Assume bin is at a fixed location
        self.path_planning_goal = (0, 0, 0)  # Goal for path planning


    def connect_components(self, builder, grasper, station):
        builder.Connect(station.GetOutputPort("spot.state_estimated"), self.get_input_port(self._spot_body_state_index))
        # builder.Connect(self.get_output_port(self._spot_commanded_state_index), station.GetInputPort("spot.desired_state"))


        # Connect the path planner to the FSM planner
        builder.Connect(self.get_output_port(self._path_planning_goal_output), self.dynamic_path_planner.GetInputPort("goal"))
        builder.Connect(self.get_output_port(self._path_planning_position_output), self.dynamic_path_planner.GetInputPort("robot_position"))
        builder.Connect(self.dynamic_path_planner.GetOutputPort("done_astar"), self.GetInputPort("path_planning_finished"))
        builder.Connect(self.GetOutputPort("use_path_planner"), self.dynamic_path_planner.GetInputPort("execute_path"))
        # builder.Connect(self.dynamic_path_planner.GetOutputPort("next_position"), self.get_input_port(self._path_planning_desired_index))

        # Connect the grasper to the FSM planner
        builder.Connect(self.GetOutputPort("request_grasp"), grasper.GetInputPort("do_grasp"))
        builder.Connect(grasper.GetOutputPort("done_grasp"), self.GetInputPort("grasp_completed"))
        
        # connect the controller to the spot input port
        # builder.Connect(self.controller.get_output_port(), station.GetInputPort("spot.desired_state"))
        # # connect the controller to the output here
        # builder.Connect(self.GetOutputPort("desired_base_position"), self.controller.GetInputPort("desired_base_position"))
        # builder.Connect(self.GetOutputPort("desired_arm_position"), self.controller.GetInputPort("desired_arm_position"))

    def get_spot_state_input_port(self):
        return self.GetInputPort("body_poses")

    def get_path_planning_finished_input_port(self):
        return self.GetInputPort("path_planning_finished")

    def _initialize_state(self, context: Context, state: State):
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        state.get_mutable_discrete_state(self._exploring).set_value([0])

    def _get_explore_completed(self, context, state):
        return self.get_path_planning_finished_input_port().Eval(context)[0]

    # def _set_target_base_position(self, context, output):
    #     (table_idx, camera_idx) = self._get_looking_inds(context)
    #     base_pose = self._camera_pos_list[table_idx, camera_idx, :].copy()
    #     if self._get_current_action(context) == Action.STEP_FORWARD:
    #         # step forward in current direction
    #         R = RotationMatrix.MakeZRotation(base_pose[2])
    #         step = R.multiply([0.2, 0, 0])
    #         base_pose[:2] += step[:2]  # ignore height in z direction
    #     if self._get_current_action(context) == Action.CARRY_BACK:
    #         base_pose = self._final_position
    #     output.SetFromVector(base_pose)


    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._state_index)).get_value()
        self.robot_state = self.get_spot_state_input_port().Eval(context)
        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()


        if mode == SpotState.IDLE:
            # Start exploring the environment
            print("State: IDLE -> EXPLORE")

            state.get_mutable_discrete_state(self.use_path_planner).set_value([0])
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.EXPLORE)

            # select a new area to explore and go to it
            new_area = self.explore_environment()

        elif mode == SpotState.EXPLORE:
            # Explore until an object is detected
            if self._get_explore_completed(context, state):
                print("Exploration completed to area.")
                if self.detect_object():
                    print("State: EXPLORE -> DETECT_OBJECT")
                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.DETECT_OBJECT)
                else:
                    print("Failed to detect object. Calculate new point for EXPLORE.")
                    new_area = self.explore_environment()
                    state.get_mutable_abstract_state(
                        int(self._state_index)
                    ).set_value(SpotState.IDLE)
            else:
                print(f"Exploring area at {self.path_planning_goal}")
                # make the controller walk the robot to the next position
                # self.next_desired_position = self.dynamic_path_planner.GetOutputPort("next_position").Eval(context)

                state.get_mutable_discrete_state(self.use_path_planner).set_value([1])
                # self.SendControllerPathOutput(context, )



        elif mode == SpotState.DETECT_OBJECT:
            if self.detected_objects:
                object_name, object_location = self.detected_objects.pop(0)
                print(f"Detected {object_name} at {object_location}")
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.APPROACH_OBJECT)
                self.current_object_location = object_location
            else:
                state.get_mutable_abstract_state(
                    int(self._state_index)
                ).set_value(SpotState.EXPLORE)

        elif mode == SpotState.APPROACH_OBJECT:
            self.approach_object(self.current_object_location)
            print("State: APPROACH_OBJECT -> GRASP_OBJECT")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.GRASP_OBJECT)

        elif mode == SpotState.GRASP_OBJECT:
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

        elif mode == SpotState.TRANSPORT_OBJECT:
            self.transport_object()
            print("State: TRANSPORT_OBJECT -> DEPOSIT_OBJECT")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.DEPOSIT_OBJECT)

        elif mode == SpotState.DEPOSIT_OBJECT:
            self.deposit_object()
            print("State: DEPOSIT_OBJECT -> RETURN_TO_IDLE")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.RETURN_TO_IDLE)

        elif mode == SpotState.RETURN_TO_IDLE:
            print("Returning to IDLE...")
            state.get_mutable_abstract_state(
                int(self._state_index)
            ).set_value(SpotState.IDLE)

    def CalcSpotState(self, context, output):
        # traj_q = context.get_mutable_abstract_state(int(self._traj_q_index)).get_value()

        # output.SetFromVector(traj_q.value(context.get_time()))
        pass

    def CalcPathPlanningGoal(self, context, output):
        output.SetFromVector(self.path_planning_goal)

    def CalcPathPlanningPosition(self, context, output):
        spot_body_pos = self.robot_state[:3]
        output.SetFromVector(spot_body_pos)

    def SendControllerPathOutput(self, context, output):
        output.SetFromVector(self.next_desired_position)

    def SendControllerArmOutput(self, context, output):
        output.SetFromVector(self.next_arm_goal)

    def detect_object(self):
        # Use Grounded SAM to detect objects
        print("Detecting objects...")
        # In the actual implementation, call the detection model here.
        success = False
        objects = []
        self.detected_objects.extend(objects)
        return success

    def approach_object(self, object_location):
        # Use A* to navigate to the object location
        print(f"Navigating to object at {object_location}")
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

    def explore_environment(self):
        # Frontier-based search for unexplored areas
        print("Exploring environment...")
        print("TODO: Actually implement this")
        # Simulate exploring a new area

        new_area = (random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0))
        self.explored_area.add(new_area)

        self.path_planning_goal = (new_area[0], new_area[1], 0.0)
        return self.path_planning_goal
