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

from .InverseKinematics import solve_ik

# q_in_front = [ 0, -0.32,  2, 0, -0.1, 0,  0]

# nominal joint angles for Spot's arm (for joint centering)
q_nominal_arm = np.array([0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0])
q_carry_arm = np.array([0.0, -1.4, 1.8, 0.0, 1.5, 0.0, 0.0])
q_drop_arm = np.array([0.0, -1.5, 1.2, 0.0, 1.5, -1.0, -1.0])
DEBUG = True

class ControllerState(Enum):
    IDLE = auto()
    REACHING_PREPICK = auto()
    REACHING_PICK = auto()
    CLOSING_GRIPPER = auto()
    MOVE_TO_CARRY_POSE = auto()
    OPENING_GRIPPER = auto()
    RETURN_TO_NOMINAL = auto()


class ControllerMission(Enum):
    STOP = 0
    PICK = auto()
    DEPOSIT = auto()

class SpotArmIKController(LeafSystem):
    def __init__(self, plant, spotplant: MultibodyPlant, bin_location, time_step=0.1, meshcat=None, use_anygrasp=False):
        super().__init__()

        self.meshcat = meshcat
        self._plant = plant
        self._spotplant = spotplant
        self.bin_location = np.array(bin_location)

        self.time_step = time_step

        self._commanded_arm_position = q_nominal_arm
        self._added_gripper_force = np.zeros(10)

        self.desired_gripper_pose = None
        self.prepick_pose = None
        self.deposit_pose = None

        self.prepick_offset = [0.00, 0, 0.2] # Temp
        if use_anygrasp:
            self.pick_offset = [0.00, 0, 0.06]
        else:
            self.pick_offset = [0.00, 0, 0.05]
        self.deposit_height = 0.7

        self.gripper_open_angle = -1.0
        self.gripper_close_angle = 0
        self.gripper_close_duration = 1.0

        self.gripper_close_start_time = 0.0
        self.gripper_open_time = 0.0

        self.gripper_open_end_time = 0.0

        self._last_solve = 0.0
        self._curr_solved_flag = 0

        self.transition_start_time = None
        self.resetPID = False

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
        self.num_attempts = 0

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
        do_arm_controller_mission = int(self.GetInputPort("do_arm_controller_mission").Eval(context)[0])
        done_grasp = bool(state.get_mutable_discrete_state(self._done_grasp).get_value()[0])

        if not do_arm_controller_mission:
            # Mission not requested, reset done_grasp and is_grasping flags
            state.get_mutable_discrete_state(self._done_grasp).set_value([0])
        elif not done_grasp:
            # do_arm_controller_mission is True, mission is not compeleted yet.
            self.run_controller_fsm(context, state)
        else:
            state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)

    def run_controller_fsm(self, context: Context, state):
        controller_mission = ControllerMission(int(self.GetInputPort("do_arm_controller_mission").Eval(context)[0]))
        controller_state = ControllerState(context.get_abstract_state(int(self._controller_state)).get_value())

        if DEBUG:
            print(f"Controller Mission: {ControllerMission(controller_mission).name}")
            print(f"Controller State: {controller_state.name}")

        commanded_base_position = self.commanded_base_position_input.Eval(context)[:3]

        curr_estimated_spot_state = self.spot_state_estimated_input.Eval(context)
        curr_base_position = curr_estimated_spot_state[:3]
        curr_q = curr_estimated_spot_state[3:10]
        curr_q_dot = curr_estimated_spot_state[13:20]

        # if controller_mission == ControllerMission.STOP:
        #     self._commanded_arm_position = curr_q
        if controller_mission == ControllerMission.PICK:
            if controller_state == ControllerState.IDLE:
                if self.transition_start_time is None:
                    self.transition_start_time = context.get_time()
                if context.get_time() < self.transition_start_time + 2.0:
                    return
                base_settled = np.allclose(curr_base_position, commanded_base_position, atol=0.1)
                if not base_settled:
                    print("Base not settled. Waiting for base to settle.")
                    return
                print("Solving IK to transition to REACHING_PREPICK")
                self.desired_gripper_pose = self.desired_gripper_pose_input.Eval(context)
                desired_gripper_pose_is_default = np.allclose(self.desired_gripper_pose.rotation().matrix(), np.eye(3)) and np.allclose(self.desired_gripper_pose.translation(), np.zeros(3))
                try:
                    self._added_gripper_force[-1] = 0.0
                    self.num_attempts += 1
                    if desired_gripper_pose_is_default:
                        print("Invalid desired gripper pose: It is the default pose.")
                        return
                    self.prepick_pose = copy(self.desired_gripper_pose)
                    self.prepick_pose.set_translation(self.desired_gripper_pose.translation() + self.prepick_offset)
                    q = solve_ik(
                        plant=self._spotplant,
                        X_WT=self.prepick_pose,
                        base_position=commanded_base_position,
                        fix_base=True,
                        max_iter=20,
                        q_current=curr_q,
                    )
                    self._commanded_arm_position = np.append(q[3:9],[self.gripper_open_angle])
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.REACHING_PREPICK)

                except AssertionError as e:
                    print(f"AssertionError caught: {e}")
                    if self.num_attempts < 10:
                        self.num_attempts = 0
                        state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.MOVE_TO_CARRY_POSE)
                        print("Gave up on grasp. Moving to carry pose to reset.")

            if controller_state == ControllerState.REACHING_PREPICK:
                # print("Current Arm Position: ", curr_q)
                # print("Commanded Arm Position: ", self._commanded_arm_position)
                # print("Difference: ", np.abs(curr_q - self._commanded_arm_position))

                if np.allclose(curr_q, self._commanded_arm_position, atol=0.1) and np.allclose(curr_q_dot, np.zeros(7), atol=0.1):
                    print("Yay! Reached the prepick pose.")

                    self.desired_gripper_pose = self.desired_gripper_pose @ RigidTransform(RotationMatrix.MakeYRotation(self.gripper_open_angle))
                    self.desired_gripper_pose.set_translation(self.desired_gripper_pose.translation() + self.pick_offset)

                    try:
                        q = solve_ik(
                            plant=self._spotplant,
                            X_WT=self.desired_gripper_pose,
                            base_position=commanded_base_position,
                            fix_base=True,
                            max_iter=20,
                            q_current=curr_q,
                            pure_vertical_movement=True,
                        )
                        self._commanded_arm_position = np.append(q[3:9],[self.gripper_open_angle])
                        self.transition_start_time = context.get_time()

                        self.prev_arm_position = curr_q
                        state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.REACHING_PICK)

                    except AssertionError as e:
                        print(f"AssertionError caught: {e}")
                        if self.num_attempts < 10:
                            self.num_attempts = 0
                            state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.MOVE_TO_CARRY_POSE)
                            print("Gave up on grasp. Moving to carry pose to reset.")

            if controller_state == ControllerState.REACHING_PICK:
                # if DEBUG and context.get_time() - self._last_print_time > 0.3:
                #     print("Current Arm Position: ", np.round(curr_q, 4))
                #     print("Commanded Arm Position: ", np.round(self._commanded_arm_position, 4))
                #     print("Difference: ", np.round(np.abs(curr_q - self._commanded_arm_position), 4))
                #     self._last_print_time = context.get_time()

                is_gripper_stopped = np.allclose(curr_q, self.prev_arm_position, atol=0.2)
                self.prev_gripper_position = curr_q[6]
                if is_gripper_stopped or (np.allclose(curr_q[:6], self._commanded_arm_position[:6], atol=0.1) and np.allclose(curr_q_dot, np.zeros(7), atol=0.1)) or context.get_time() - self.transition_start_time > 1.0:
                    print("Reached the PICK pose. Closing gripper")
                    print("CLOSING GRIPPER TIME: ", context.get_time())

                    self.gripper_close_start_time = context.get_time()
                    self.gripper_close_end_time = self.gripper_close_start_time + self.gripper_close_duration
                    elapsed_time = context.get_time() - self.gripper_close_start_time
                    # self._commanded_arm_position[6] = gripper_angle
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.CLOSING_GRIPPER)
                    self.prev_gripper_position = curr_q[6]


            if controller_state == ControllerState.CLOSING_GRIPPER:
                # if DEBUG and context.get_time() - self._last_print_time > 0.5:
                #     # print("Current Gripper Position: ", curr_q[6])
                #     # print("Commanded Gripper Position: ", self._commanded_arm_position[6])
                #     # print("Difference: ", np.abs(curr_q[6] - self._commanded_arm_position[6]))
                #     # print("Current Arm Position: ", np.round(curr_q, 4))
                #     # print("Commanded Arm Position: ", np.round(self._commanded_arm_position, 4))
                #     # print("Difference: ", np.round(np.abs(curr_q - self._commanded_arm_position), 4))
                #     self._last_print_time = context.get_time()
                elapsed_time = context.get_time() - self.gripper_close_start_time
                alpha = elapsed_time / self.gripper_close_duration
                gripper_angle = (1-alpha)*curr_q[6] + alpha*self.gripper_close_angle
                self._commanded_arm_position[6] = gripper_angle
                self._added_gripper_force[-1] = alpha * 15.0

                is_gripper_closed = np.abs(curr_q[6] - self.prev_gripper_position) < 0.01
                self.prev_gripper_position = curr_q[6]
                if context.get_time() > self.gripper_close_end_time or is_gripper_closed:
                    print("Gripper closed. Moving to carry arm pose.")
                    self._commanded_arm_position = curr_q
                    self.transition_start_time = context.get_time()
                    self.resetPID = True # Test
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.MOVE_TO_CARRY_POSE)

            if controller_state == ControllerState.MOVE_TO_CARRY_POSE:
                if self.transition_start_time is None:
                    self.transition_start_time = context.get_time()

                self.resetPID = False # Test

                elapsed_time = context.get_time() - self.transition_start_time
                transition_duration = 1.0  # Time in seconds to complete transition
                timeout = 2.0

                self._added_gripper_force[-1] = 15.0

                if elapsed_time < transition_duration:
                    # Linear interpolation from current position to nominal position
                    alpha = elapsed_time / transition_duration
                    self._commanded_arm_position = (1 - alpha) * curr_q + alpha * q_carry_arm
                    # print(f"Transitioning to carry pose: {self._commanded_arm_position}")
                elif elapsed_time >= transition_duration + timeout:
                    print("Gave up on carry pose. Ready i guess?")
                    # self._commanded_arm_position = curr_q
                    self.transition_start_time = None  # Reset the transition time for future use
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                    state.get_mutable_discrete_state(self._done_grasp).set_value([1])
                else:
                    # Finalize the transition once time has elapsed
                    self._commanded_arm_position = q_carry_arm
                    if np.allclose(curr_q, q_carry_arm, atol=0.5) and np.allclose(curr_q_dot, np.zeros(7), atol=0.5):
                        print("Returned to carry arm pose. Ready to transport")
                        self.transition_start_time = None  # Reset the transition time for future use
                        state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                        state.get_mutable_discrete_state(self._done_grasp).set_value([1])

        if controller_mission == ControllerMission.DEPOSIT:
            if controller_state == ControllerState.IDLE:
                print("Dropping object in the bin. Opening gripper.")
                # self.deposit_pose = self.calculate_deposit_pose(commanded_base_position)
                # self.deposit_pose = self.deposit_pose @ RigidTransform(RotationMatrix.MakeYRotation(curr_q[6]))

                # self._commanded_arm_position = q_drop_arm
                self._commanded_arm_position[6] = self.gripper_open_angle
                self._added_gripper_force[-1] = 0.0
                self.gripper_open_end_time = context.get_time() + 2.0
                state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.OPENING_GRIPPER)

            if controller_state == ControllerState.OPENING_GRIPPER:
                if context.get_time() > self.gripper_open_end_time:
                    print("Gripper open. Returning to nominal arm pose.")

                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.RETURN_TO_NOMINAL)
                    self.transition_start_time = context.get_time()

            if controller_state == ControllerState.RETURN_TO_NOMINAL:

                # self.transition_start_time = None  # Reset the transition time for future use
                # state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                # state.get_mutable_discrete_state(self._done_grasp).set_value([1])
                # self._commanded_arm_position = curr_q
                # WHO CARES ABOUT THE NOMINAL POSE? JUST DROP THE OBJECT AND BE DONE WITH IT
                if self.transition_start_time is None:
                    self.transition_start_time = context.get_time()

                elapsed_time = context.get_time() - self.transition_start_time
                transition_duration = 1.8  # Time in seconds to complete transition
                timeout = 2

                if elapsed_time < transition_duration:
                    # Linear interpolation from current position to nominal position
                    alpha = elapsed_time / transition_duration
                    self._commanded_arm_position = (1 - alpha) * curr_q + alpha * q_carry_arm
                    # print(f"Transitioning to carry pose: {self._commanded_arm_position}")
                elif elapsed_time >= transition_duration + timeout:
                    print("Gave up on nominal pose. Ready i guess?")
                    self.transition_start_time = None  # Reset the transition time for future use
                    state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                    state.get_mutable_discrete_state(self._done_grasp).set_value([1])
                    self._commanded_arm_position = curr_q
                else:
                    self._commanded_arm_position = q_carry_arm
                    if np.allclose(curr_q, q_carry_arm, atol=0.2) and np.allclose(curr_q_dot, np.zeros(7), atol=0.2):
                        print("Returned to nominal arm pose. Ready for next mission.")
                        self.transition_start_time = None  # Reset the transition time for future use
                        state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                        state.get_mutable_discrete_state(self._done_grasp).set_value([1])
                        self._commanded_arm_position = curr_q


            # if controller_state == ControllerState.REACHING_DEPOSIT:
            #     print("Waiting for the arm to reach the deposit pose.")
            #     if np.allclose(curr_q[:6], self._commanded_arm_position[:6], atol=0.1) and np.allclose(curr_q_dot, np.zeros(7), atol=0.1):
            #         print("Reached the deposit pose!!!")
                    # state.get_mutable_abstract_state(int(self._controller_state)).set_value(ControllerState.IDLE)
                    # state.get_mutable_discrete_state(self._done_grasp).set_value([1])


    def calculate_deposit_pose(self, reference_position):
        reference_position = np.array(reference_position[:2].tolist() + [0.0])
        direction_to_bin = self.bin_location - reference_position
        unit_vector_to_bin = direction_to_bin / np.linalg.norm(direction_to_bin)

        z_axis = unit_vector_to_bin
        x_axis = np.array([0, 0, -1])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        R = RotationMatrix(np.column_stack((x_axis, y_axis, z_axis)))
        translation = self.bin_location.copy()
        translation[2] = self.deposit_height

        return RigidTransform(R, translation)
