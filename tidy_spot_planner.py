import random
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    AbstractValue,
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

        # Spot's internal planning states
        self._state_index = self.DeclareAbstractState(AbstractValue.Make(SpotState.IDLE))
        # self._traj_X_G_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePose()))
        # self._traj_wsg_index = self.DeclareAbstractState(AbstractValue.Make(PiecewisePolynomial()))

        self._mode_index = self.DeclareAbstractState(
            AbstractValue.Make(SpotState.IDLE)
        )
        self._times_index = self.DeclareAbstractState(
            AbstractValue.Make({"initial": 0.0})
        )
        self._attempts_index = self.DeclareDiscreteState(1)

        # Input ports
        self._spot_body_state_index = self.DeclareVectorInputPort("body_poses", 20).get_index()
        #self._grasp_input_index = self.DeclareAbstractInputPort("grasp_selection", AbstractValue.Make((np.inf, RigidTransform()))).get_index()
        #self._wsg_state_index = self.DeclareVectorInputPort("wsg_state", 2).get_index()

        # Output ports
        # self.DeclareAbstractOutputPort("X_WG", lambda: AbstractValue.Make(RigidTransform()), self.CalcGripperPose)
        self._spot_commanded_state_index = self.DeclareVectorOutputPort("spot_commanded_state", 20, self.CalcSpotState).get_index()

        self._rng = np.random.default_rng()

        self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

        self.detected_objects = []
        self.explored_area = set()  # Grid to track explored areas
        self.bin_location = (0, 0)  # Assume bin is at a fixed location

    def connect_components(self, builder, station):
        builder.Connect(station.GetOutputPort("spot.state_estimated"), self.get_input_port(self._spot_body_state_index))
        builder.Connect(self.get_output_port(self._spot_commanded_state_index), station.GetInputPort("spot.desired_state"))

    def Update(self, context, state):
        mode = context.get_abstract_state(int(self._state_index)).get_value()

        current_time = context.get_time()
        times = context.get_abstract_state(int(self._times_index)).get_value()


        if mode == SpotState.IDLE:
            # Start exploring the environment
            print("State: IDLE -> EXPLORE")

            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(SpotState.EXPLORE)

        elif mode == SpotState.EXPLORE:
            # Explore until an object is detected
            new_area = self.explore_environment()
            if self.detect_object():
                print("State: EXPLORE -> DETECT_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(SpotState.DETECT_OBJECT)
            else:
                print(f"Explored area at {new_area}")

        elif mode == SpotState.DETECT_OBJECT:
            if self.detected_objects:
                object_name, object_location = self.detected_objects.pop(0)
                print(f"Detected {object_name} at {object_location}")
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(SpotState.APPROACH_OBJECT)
                self.current_object_location = object_location
            else:
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(SpotState.EXPLORE)

        elif mode == SpotState.APPROACH_OBJECT:
            self.approach_object(self.current_object_location)
            print("State: APPROACH_OBJECT -> GRASP_OBJECT")
            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(SpotState.GRASP_OBJECT)

        elif mode == SpotState.GRASP_OBJECT:
            if self.grasp_object():
                print("State: GRASP_OBJECT -> TRANSPORT_OBJECT")
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(SpotState.TRANSPORT_OBJECT)
            else:
                print("Failed to grasp object. Returning to EXPLORE.")
                state.get_mutable_abstract_state(
                    int(self._mode_index)
                ).set_value(SpotState.EXPLORE)

        elif mode == SpotState.TRANSPORT_OBJECT:
            self.transport_object()
            print("State: TRANSPORT_OBJECT -> DEPOSIT_OBJECT")
            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(SpotState.DEPOSIT_OBJECT)

        elif mode == SpotState.DEPOSIT_OBJECT:
            self.deposit_object()
            print("State: DEPOSIT_OBJECT -> RETURN_TO_IDLE")
            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(SpotState.RETURN_TO_IDLE)

        elif mode == SpotState.RETURN_TO_IDLE:
            print("Returning to IDLE...")
            state.get_mutable_abstract_state(
                int(self._mode_index)
            ).set_value(SpotState.IDLE)

    def CalcSpotState(self, context, output):
        # traj_q = context.get_mutable_abstract_state(int(self._traj_q_index)).get_value()

        # output.SetFromVector(traj_q.value(context.get_time()))
        pass

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
        # Simulate exploring a new area
        new_area = (random.randint(1, 10), random.randint(1, 10))
        self.explored_area.add(new_area)
        return new_area

    def run_state_machine(self):
        # Main loop for FSM
        while True:
            if mode == SpotState.IDLE:
                # Start exploring the environment
                print("State: IDLE -> EXPLORE")
                mode = SpotState.EXPLORE

            elif mode == SpotState.EXPLORE:
                # Explore until an object is detected
                new_area = self.explore_environment()
                if self.detect_object():
                    print("State: EXPLORE -> DETECT_OBJECT")
                    mode = SpotState.DETECT_OBJECT
                else:
                    print(f"Explored area at {new_area}")

            elif mode == SpotState.DETECT_OBJECT:
                if self.detected_objects:
                    object_name, object_location = self.detected_objects.pop(0)
                    print(f"Detected {object_name} at {object_location}")
                    mode = SpotState.APPROACH_OBJECT
                    self.current_object_location = object_location
                else:
                    mode = SpotState.EXPLORE

            elif mode == SpotState.APPROACH_OBJECT:
                self.approach_object(self.current_object_location)
                print("State: APPROACH_OBJECT -> GRASP_OBJECT")
                mode = SpotState.GRASP_OBJECT

            elif mode == SpotState.GRASP_OBJECT:
                if self.grasp_object():
                    print("State: GRASP_OBJECT -> TRANSPORT_OBJECT")
                    mode = SpotState.TRANSPORT_OBJECT
                else:
                    print("Failed to grasp object. Returning to EXPLORE.")
                    mode = SpotState.EXPLORE

            elif mode == SpotState.TRANSPORT_OBJECT:
                self.transport_object()
                print("State: TRANSPORT_OBJECT -> DEPOSIT_OBJECT")
                mode = SpotState.DEPOSIT_OBJECT

            elif mode == SpotState.DEPOSIT_OBJECT:
                self.deposit_object()
                print("State: DEPOSIT_OBJECT -> RETURN_TO_IDLE")
                mode = SpotState.RETURN_TO_IDLE

            elif mode == SpotState.RETURN_TO_IDLE:
                print("Returning to IDLE...")
                mode = SpotState.IDLE

