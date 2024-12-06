import time
import numpy as np

from pydrake.all import (
    AbstractValue,
    BasicVector,
    Context,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)

from .inverse_kinematics import q_nominal_arm, solve_ik

q_in_front = [ 0, -0.32,  2, 0, -0.1, 0,  0]
PRINT = False

class SpotArmIKController(LeafSystem):
    """Given a desire pose, compute the joint angles for Spot's arm.

    Currently, for debugging purpose, it is configured such that, it
    will try to reach the banana whenever possible (i.e. if IK succeeds).
    Otherwise, it will fold the arm back to the nominal pose.
    """

    def __init__(self, plant: MultibodyPlant, old_nav: bool = True, time_step=0.1):
        super().__init__()

        self._plant = plant
        self.time_step = time_step

        self._commanded_arm_position = q_nominal_arm
        self.gripper_offset = [0, 0, 0.18]
        self._last_solve = 0.0
        self._curr_solved_flag = 0

        self.DeclareAbstractInputPort(
            "desired_pose", AbstractValue.Make(RigidTransform())
        )
        self.DeclareVectorInputPort("spot.state_estimated",20)
        self.DeclareVectorInputPort("commanded_base_position", 10 if old_nav else 3)

        self.DeclareVectorOutputPort("desired_spot_arm_position", 7, self.UpdateDesiredArmPosition)

        # Periodic update for trajectory generation
        self._state_update_event = self.DeclarePeriodicUnrestrictedUpdateEvent(self.time_step, 0.0, self.UpdateGrippingSkill)
        
        if PRINT: 
            self._last_print = 0.0
            self.solving_num = 0

        self.curr_demanding_pose = RigidTransform()
        self.gripping_state = "Waiting for Input"
        #TODO maybe resting state other than q_nominal_arm

        # FSM Handling

        # Input ports
        self._do_grasp_input = self.DeclareVectorInputPort("do_grasp", 1)

        # Internal states
        self._done_grasp = self.DeclareDiscreteState(1)
        self._is_grasping = self.DeclareDiscreteState(1)  # 0: Not grasping, 1: Grasping

        # Output ports
        self.DeclareStateOutputPort("done_grasp", self._done_grasp) # TODO: Add success/fail flag
        self.DeclareStateOutputPort("is_grasping", self._is_grasping)

    def connect_ports(self, builder, spot_station, base_position_commander, grasp_selector): # Add modules as parameters to be connected to the PositionCombiner
        builder.Connect(
            grasp_selector.GetOutputPort("grasp_selection"),
            self.GetInputPort("desired_pose"),
        )
        builder.Connect(
            spot_station.GetOutputPort("spot.state_estimated"),
            self.GetInputPort("spot.state_estimated"),
        )
        builder.Connect(
            base_position_commander.GetOutputPort("desired_base_position"),
            self.GetInputPort("commanded_base_position"),
        )


    def get_desired_pose_input_port(self):
        return self.GetInputPort("desired_pose")

    def get_commanded_base_position_input_port(self):
        return self.GetInputPort("commanded_base_position")

    def get_estimated_spot_state_input_port(self):
        return self.GetInputPort("spot.state_estimated")

    def UpdateDesiredArmPosition(self, context: Context, output):
        output.SetFromVector(self._commanded_arm_position)

    def UpdateGrippingSkill(self, context: Context, state):
        state_do_grasp = self.GetInputPort("do_grasp").Eval(context)[0]
        state_done_grasp = state.get_mutable_discrete_state(self._done_grasp).get_value()[0]
        state_is_grasping = state.get_mutable_discrete_state(self._is_grasping).get_value()[0]

        if not state_do_grasp:
            state.get_mutable_discrete_state(self._done_grasp).set_value([0])
        elif state_do_grasp and not state_done_grasp:
            state.get_mutable_discrete_state(self._is_grasping).set_value([1])
            # Call actual Grasping Skill
            self.run_grasping_algorithm(context: Context, state)
        elif state_do_grasp and state_done_grasp:
            state.get_mutable_discrete_state(self._is_grasping).set_value([0])
            state.get_mutable_discrete_state(self._done_grasp).set_value([1])
        else:
            print("unexpected state in ArmIKController")

    def run_grasping_algorithm(self, context: Context, state):
        if PRINT and context.get_time() - self._last_print > 0.1:
            print(self.gripping_state)
            self._last_print = context.get_time()

        base_position = self.get_commanded_base_position_input_port().Eval(context)[:3]
        desired_pose = self.EvalAbstractInput(context, 0).get_value()

        curr_estimated_spot_state = self.get_estimated_spot_state_input_port().Eval(context)
        curr_q = curr_estimated_spot_state[3:10]
        curr_base_position = curr_estimated_spot_state[:3]

        if self.gripping_state == "Waiting for Input":
            if (self.curr_demanding_pose == desired_pose and self._curr_solved_flag == 1) or self.curr_demanding_pose == RigidTransform(): 
                if PRINT:
                    self.solving_num = 0
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position
            else:
                self.gripping_state = "Solving IK"

        if self.gripping_state == "Solving IK":
            self.curr_demanding_pose = desired_pose
            self._curr_solved_flag == 0

            if PRINT: 
                self.solving_num += 1
                print(self.solving_num)

            try:
                prepare_pose = self.curr_demanding_pose
                prepare_pose.set_translation(self.curr_demanding_pose.translation() + self.gripper_offset + [0, 0, 0.2])
                q = solve_ik(
                    plant=self._plant,
                    context=self._plant.CreateDefaultContext(),
                    # assuming that we'd like to reach from above
                    X_WT=prepare_pose,
                    base_position=base_position,
                    fix_base=True,
                    max_iter=5,
                    # to accelerate solving, we give an arm position in front of the robot as optimization start
                    #q_current=q_in_front,
                    q_current=q_nominal_arm,
                )
                self._commanded_arm_position = q[3:10]
                self.gripping_state = "Reaching pose"
                print(self._commanded_arm_position) if PRINT
                self._curr_solved_flag = 1

            except AssertionError:
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position
            finally:
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position

        elif (self.gripping_state == "Reaching pose" and 
                np.allclose(curr_q, self._commanded_arm_position, atol=0.01) and 
                np.allclose(curr_base_position, base_position, atol=0.01)):
            q_arm = np.append(self._commanded_arm_position[:6],[-1.0])
            self._commanded_arm_position = q_arm
            self.gripping_state = "Opening Gripper"
            self._last_solve = context.get_time()
            # I know this is stupid, but I had to structure this like this for testing purposes
            self._commanded_arm_position = self._commanded_arm_position
        elif self.gripping_state == "Reaching pose":
            # I know this is stupid, but I had to structure this like this for testing purposes
            self._commanded_arm_position = self._commanded_arm_position

        elif self.gripping_state == "Opening Gripper" and context.get_time() - self._last_solve < 0.1:
            # I know this is stupid, but I had to structure this like this for testing purposes
            self._commanded_arm_position = self._commanded_arm_position
        elif self.gripping_state == "Opening Gripper":
            try:
                gripping_pose = self.curr_demanding_pose
                gripping_pose.set_translation(self.curr_demanding_pose.translation() + self.gripper_offset)
                q = solve_ik(
                    plant=self._plant,
                    context=self._plant.CreateDefaultContext(),
                    X_WT=gripping_pose,
                    base_position=base_position,
                    fix_base=True,
                    max_iter=1,
                    q_current=self._commanded_arm_position,
                )
                self._commanded_arm_position = np.append(q[3:9],[-1.0])
                self.gripping_state = "Reaching Closing Pose"
                self._last_solve = context.get_time()

            except AssertionError:
                self._commanded_arm_position = self._commanded_arm_position
            finally:
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position

        elif self.gripping_state == "Reaching Closing Pose" and context.get_time() - self._last_solve < 0.4:
            # I know this is stupid, but I had to structure this like this for testing purposes
            self._commanded_arm_position = self._commanded_arm_position
        elif self.gripping_state == "Reaching Closing Pose":
            q_arm = np.append(self._commanded_arm_position[:6],[0.0])
            self.gripping_state = "Closing Gripper"
            self._commanded_arm_position = q_arm

        elif self.gripping_state == "Closing Gripper":
            # give it 0.1sec to close the gripper
            if context.get_time() - self._last_solve < 1:
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position
            else: 
                self.gripping_state = "Waiting for Input"
                self._commanded_arm_position = q_nominal_arm
                # I know this is stupid, but I had to structure this like this for testing purposes
                self._commanded_arm_position = self._commanded_arm_position

        else: 
            if context.get_time() - self._last_solve < 0.25:
                print("Huston we have a problem")