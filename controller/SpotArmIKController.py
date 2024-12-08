import time
import numpy as np
from enum import Enum, auto
from copy import copy

from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
)

from .InverseKinematics import solve_ik, solve_ik_downwards_only

# q_in_front = [ 0, -0.32,  2, 0, -0.1, 0,  0]

# nominal joint angles for Spot's arm (for joint centering)
q_nominal_arm = np.array([0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0])
DEBUG = True
VIZUALIZE_POSES = True

class ControllerState(Enum):
    IDLE = auto()
    REACHING_PREPICK = auto()
    OPENING_GRIPPER = auto()
    REACHING_PICK = auto()
    CLOSING_GRIPPER = auto()
    RETURN_TO_NOMINAL_ARM = auto()

class ControllerMission(Enum):
    STOP = 0
    PICK = auto()
    DEPOSIT = auto()

class SpotArmIKController(LeafSystem):
    def __init__(self, plant, spotplant: MultibodyPlant, time_step=0.1, meshcat=None):
        super().__init__()

        self.meshcat = meshcat
        self._plant = plant
        self._spotplant = spotplant

        self.time_step = time_step

        self._commanded_arm_position = q_nominal_arm
        self._added_gripper_force = np.zeros(10)

        self.desired_gripper_pose = None
        self.prepick_pose = None

        self.prepick_offset = [0.01, 0, 0.2] # Temp
        self.gripper_offset = [0.01, 0, 0.075]
        self.gripper_open_angle = -1.0
        self.gripper_close_duration = 5.0

        self.gripper_close_start_time = 0.0
        self.gripper_open_time = 0.0

        self._last_solve = 0.0
        self._curr_solved_flag = 0

        # Input ports
        self.desired_gripper_pose_input = self.DeclareAbstractInputPort("desired_gripper_pose", AbstractValue.Make(RigidTransform()))
        self.spot_state_estimated_input = self.DeclareVectorInputPort("spot.state_estimated",20)
        self.commanded_base_position_input = self.DeclareVectorInputPort("commanded_base_position", 3)

        self._do_arm_controller_mission_input = self.DeclareVectorInputPort("do_arm_controller_mission", 1) # 0: Stop, 1: Pick, 2: Deposit

        # Internal states
        self._controller_state = self.DeclareAbstractState(AbstractValue.Make(ControllerState.IDLE))
        self._prev_controller_state = self.DeclareAbstractState(AbstractValue.Make(ControllerState.IDLE)) # Flag to ensure IK is solved only once per attempt. Initialize with 0 (False)

        self._done_grasp = self.DeclareDiscreteState(1)
        # self._is_grasping = self.DeclareDiscreteState(1)  # 0: Not grasping, 1: Grasping

        # Output ports
        self.DeclareVectorOutputPort("desired_spot_arm_position", 7, self.UpdateDesiredArmPosition)
        self.DeclareVectorOutputPort("added_gripper_force", 10, self.UpdateAddedGripperForce)

        self.DeclareStateOutputPort("done_grasp", self._done_grasp) # TODO: Add success/fail flag
        # self.DeclareStateOutputPort("is_grasping", self._is_grasping)

        self._state_update_event = self.DeclarePeriodicUnrestrictedUpdateEvent(self.time_step, 0.0, self.Update)

        # DEBUG VARIABLES
        self._last_print_time = 0.0
        self.solving_num = 0

    def connect_components(self, builder, station, navigator, grasp_selector):
        builder.Connect(
            grasp_selector.GetOutputPort("grasp_selection"),
            self.GetInputPort("desired_gripper_pose"),
        )
        builder.Connect(
            station.GetOutputPort("spot.state_estimated"),
            self.GetInputPort("spot.state_estimated"),
        )
        builder.Connect(
            navigator.GetOutputPort("spot_commanded_position"),
            self.GetInputPort("commanded_base_position"),
        )
        builder.Connect(
            self.GetOutputPort("added_gripper_force"),
            station.GetInputPort("spot.added_actuation_force"),
        )

    def UpdateDesiredArmPosition(self, context: Context, output):
        output.SetFromVector(self._commanded_arm_position)

    def UpdateAddedGripperForce(self, context: Context, output):
        output.SetFromVector(self._added_gripper_force)

    def Update(self, context: Context, state):
        curr_estimated_spot_state = self.spot_state_estimated_input.Eval(context)
        curr_base_position = curr_estimated_spot_state[:3]
        curr_q = curr_estimated_spot_state[3:10]
        curr_q_dot = curr_estimated_spot_state[13:20]

        # if context.get_time() > 1:
        #     if context.get_time() - self.gripper_close_time > 3:
        #         self._commanded_arm_position = np.append(self._commanded_arm_position[:6], [0.0])
        #     else:
        #         self._commanded_arm_position = np.append(self._commanded_arm_position[:6], [-1.0])

        # return


        do_arm_controller_mission = bool(self.GetInputPort("do_arm_controller_mission").Eval(context)[0])

        done_grasp_state = state.get_mutable_discrete_state(self._done_grasp)
        done_grasp = bool(done_grasp_state.get_value()[0])

        if not do_arm_controller_mission:
            # Pick or deposit not requested, reset done_grasp and is_grasping flags
            done_grasp_state.set_value([0])
        elif not done_grasp:
            # do_grasp is True, grasp is not compeleted yet.
            self.run_grasping_algorithm(context, state)
        else:
            state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.STOP)

    def run_grasping_algorithm(self, context: Context, state):
        controller_mission = ControllerMission(int(self.GetInputPort("do_arm_controller_mission").Eval(context)[0]))
        controller_state = ControllerState(context.get_abstract_state(int(self._controller_state)).get_value())

        # if DEBUG:
        #     print(f"Controller Mission: {ControllerMission(controller_mission).name}")
        #     print(f"Controller State: {controller_state.name}")

        commanded_base_position = self.commanded_base_position_input.Eval(context)[:3]

        curr_estimated_spot_state = self.spot_state_estimated_input.Eval(context)
        curr_base_position = curr_estimated_spot_state[:3]
        curr_q = curr_estimated_spot_state[3:10]
        curr_q_dot = curr_estimated_spot_state[13:20]

        if controller_mission == ControllerMission.STOP:
            state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
            return

        if controller_mission == ControllerMission.PICK:
            if controller_state == ControllerState.IDLE:
                print("Solving IK to transition to REACHING_PREPICK")
                self.desired_gripper_pose = self.desired_gripper_pose_input.Eval(context)
                desired_gripper_pose_is_default = np.allclose(self.desired_gripper_pose.rotation().matrix(), np.eye(3)) and np.allclose(self.desired_gripper_pose.translation(), np.zeros(3))

                try:
                    if desired_gripper_pose_is_default:
                        print("Invalid desired gripper pose: It is the default pose.")
                        return
                    self.prepick_pose = copy(self.desired_gripper_pose)
                    self.prepick_pose.set_translation(self.desired_gripper_pose.translation() + self.prepick_offset)
                    q = solve_ik(
                        plant=self._spotplant,
                        context=self._spotplant.CreateDefaultContext(),
                        X_WT=self.prepick_pose,
                        base_position=commanded_base_position,
                        fix_base=True,
                        max_iter=20,
                        q_current=curr_q,
                    )
                    self._commanded_arm_position = np.append(q[3:9],[self.gripper_open_angle])
                    # print("Prepick commanded", self._commanded_arm_position)
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.REACHING_PREPICK)

                except AssertionError as e:
                    print(f"AssertionError caught: {e}")
                    # Handle the assertion failure (e.g., reset pose or log an error)

            if controller_state == ControllerState.REACHING_PREPICK:
                # print("Current Arm Position: ", curr_q)
                # print("Commanded Arm Position: ", self._commanded_arm_position)
                # print("Difference: ", np.abs(curr_q - self._commanded_arm_position))

                if np.allclose(curr_q, self._commanded_arm_position, atol=0.1) and np.allclose(curr_q_dot, np.zeros(7), atol=0.1):
                    print("Yay! Reached the prepick pose.")

                    self.desired_gripper_pose = self.desired_gripper_pose @ RigidTransform(RotationMatrix.MakeYRotation(self.gripper_open_angle))
                    self.desired_gripper_pose.set_translation(self.desired_gripper_pose.translation() + self.gripper_offset)

                    try:
                        q = solve_ik_downwards_only(
                            plant=self._spotplant,
                            context=self._spotplant.CreateDefaultContext(),
                            X_WT=self.desired_gripper_pose,
                            base_position=commanded_base_position,
                            fix_base=True,
                            max_iter=20,
                            q_current=curr_q,
                        )
                        self._commanded_arm_position = np.append(q[3:9],[self.gripper_open_angle])
                        state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.REACHING_PICK)

                        self._last_print_time = context.get_time()

                    except AssertionError as e:
                        print(f"AssertionError caught: {e}")

            if controller_state == ControllerState.REACHING_PICK:
                # if DEBUG and context.get_time() - self._last_print_time > 0.3:
                # #     # print("Current Arm Position: ", np.round(curr_q, 4))
                # #     # print("Commanded Arm Position: ", np.round(self._commanded_arm_position, 4))
                # #     print("Difference: ", np.round(np.abs(curr_q - self._commanded_arm_position), 4))
                #     self._last_print_time = context.get_time()

                if np.allclose(curr_q[:6], self._commanded_arm_position[:6], atol=0.1) and np.allclose(curr_q_dot, np.zeros(7), atol=0.1):
                    print("Reached the PICK pose. Closing gripper")
                    print("CLOSING GRIPPER TIME: ", context.get_time())

                    self.gripper_close_start_time = context.get_time()
                    self.gripper_close_end_time = self.gripper_close_start_time + self.gripper_close_duration
                    self._added_gripper_force[-1] = 3.0
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.CLOSING_GRIPPER)


            if controller_state == ControllerState.CLOSING_GRIPPER:
                if DEBUG and context.get_time() - self._last_print_time > 0.5:
                    # print("Current Gripper Position: ", curr_q[6])
                    # print("Commanded Gripper Position: ", self._commanded_arm_position[6])
                    # print("Difference: ", np.abs(curr_q[6] - self._commanded_arm_position[6]))
                    # print("Current Arm Position: ", np.round(curr_q, 4))
                    # print("Commanded Arm Position: ", np.round(self._commanded_arm_position, 4))
                    # print("Difference: ", np.round(np.abs(curr_q - self._commanded_arm_position), 4))
                    self._last_print_time = context.get_time()

                # if context.get_time() - self.gripper_close_time > 1:
                #     print("Gripper closed. Returning to nominal arm pose.")
                #     self._commanded_arm_position = q_nominal_arm
                #     state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.RETURN_TO_NOMINAL_ARM)

            if controller_state == ControllerState.RETURN_TO_NOMINAL_ARM:
                if np.allclose(curr_q, q_nominal_arm, atol=0.03) and np.allclose(curr_q_dot, np.zeros(7), atol=0.03):
                    print("Returned to nominal arm pose. Ready to transport")
                    # state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                    # done_grasp_state.set_value([1])

        if controller_mission == ControllerMission.DEPOSIT:
            pass
