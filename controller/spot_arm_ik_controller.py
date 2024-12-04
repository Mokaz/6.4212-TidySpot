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


class SpotArmIKController(LeafSystem):
    """Given a desire pose, compute the joint angles for Spot's arm.

    Currently, for debugging purpose, it is configured such that, it
    will try to reach the banana whenever possible (i.e. if IK succeeds).
    Otherwise, it will fold the arm back to the nominal pose.
    """

    def __init__(
        self, plant: MultibodyPlant, enabled: bool = True):
        super().__init__()

        self._plant = plant
        self.DeclareAbstractInputPort(
            "desired_pose", AbstractValue.Make(RigidTransform())
        )
        self.DeclareVectorInputPort("spot.state_estimated",20)
        self.DeclareVectorInputPort("base_position", 10)
        self.DeclareVectorOutputPort("desired_spot_arm_position", 7, self._solve_ik)
        self._last_solve = time.time()
        self._last_state = q_nominal_arm
        #TODO maybe resting state other than q_nominal_arm

        self._enabled = enabled

    def get_desired_pose_input_port(self):
        return self.GetInputPort("desired_pose")

    def get_base_position_input_port(self):
        return self.GetInputPort("base_position")

    def _solve_ik(self, context: Context, output: BasicVector):
        # because IK is a time-consuming process, we only want to do it once per second
        # in actual application, we shouldn't be solving IK in real time...
        # we should consider using RRT or something similar to plan the motion
        if not self._enabled or time.time() - self._last_solve < 0.25:
            output.SetFromVector(self._last_state)
            return
        self._last_solve = time.time()

        base_position = self.get_base_position_input_port().Eval(context)[:3]
        desired_pose = self.EvalAbstractInput(context, 0).get_value()
        # assuming that this is the banana pose, and we'd like to reach it from above
        desired_pose.set_translation(desired_pose.translation() + [0, 0, 0.2])
        try:
            q = solve_ik(
                plant=self._plant,
                context=self._plant.CreateDefaultContext(),
                X_WT=desired_pose,
                base_position=base_position,
                fix_base=True,
                max_iter=1,
            )
            self._last_state = q[3:10]

        except AssertionError:
            self._last_state = q_nominal_arm
        finally:
            output.SetFromVector(self._last_state)

    def _gripping_action(self, context: Context, output: BasicVector):
        if time.time() - self._last_print < 0.1:
            output.SetFromVector(self._last_state)
        else:
            print(self.gripping_state)
            self._last_print = time.time()
        self._last_solve = time.time()

        base_position = self.get_base_position_input_port().Eval(context)[:3]
        desired_pose = self.EvalAbstractInput(context, 0).get_value()
        # assuming that this is the banana pose, and we'd like to reach it from above
        desired_pose.set_translation(desired_pose.translation() + [0, 0, 0.2])

        # where do I get the joint positions from? another input?
        curr_q = self.get_estimated_spot_state_input_port().Eval(context)[3:10]
        
        if self.gripping_state == "Waiting for Input":
            self.curr_demanding_pose = desired_pose
            try:
                q = solve_ik(
                    plant=self._plant,
                    context=self._plant.CreateDefaultContext(),
                    X_WT=self.curr_demanding_pose,
                    base_position=base_position,
                    fix_base=True,
                    max_iter=1,
                )
                self._last_state = q[3:10]
                self.gripping_state = "Reaching pose"

            except AssertionError:
                self._last_state = q_nominal_arm
            finally:
                output.SetFromVector(self._last_state)

        elif self.gripping_state == "Reaching pose" and not np.allclose(curr_q, self._last_state, atol=0.001):
            output.SetFromVector(self._last_state)
        elif self.gripping_state == "Reaching pose" and np.allclose(curr_q, self._last_state, atol=0.001):
            q_arm = np.append(self._last_state[:6],[-0.5])
            self._last_state = q_arm
            self.gripping_state = "Opening Gripper"
            output.SetFromVector(self._last_state)

        elif self.gripping_state == "Opening Gripper" and not np.allclose(curr_q, self._last_state, atol=0.001):
            print(curr_q,self._last_state)
            output.SetFromVector(self._last_state)
        elif self.gripping_state == "Opening Gripper" and np.allclose(curr_q, self._last_state, atol=0.001):
            self.curr_demanding_pose.set_translation(self.curr_demanding_pose.translation() - [0, 0, 0.2])
            try:
                q = solve_ik(
                    plant=self._plant,
                    context=self._plant.CreateDefaultContext(),
                    X_WT=self.curr_demanding_pose,
                    base_position=base_position,
                    fix_base=True,
                    max_iter=1,
                )
                self._last_state = np.append(q[3:9],[-0.5])

            except AssertionError:
                self._last_state = q_nominal_arm
                print("Huston we have a problem")
            finally:
                self.gripping_state = "Reaching Closing Pose"
                output.SetFromVector(self._last_state)

        elif self.gripping_state == "Reaching Closing Pose" and not np.allclose(curr_q, self._last_state, atol=0.001):
            output.SetFromVector(self._last_state)
        elif self.gripping_state == "Reaching Closing Pose" and np.allclose(curr_q, self._last_state, atol=0.001):
            q_arm = np.append(self._last_state[:6],[0.0])
            self.gripping_state = "Closing Gripper"
            self._last_state = q_arm

        elif self.gripping_state == "Closing Gripper":
            # give it 0.25sec to close the gripper
            if not self._enabled or time.time() - self._last_solve < 0.25:
                output.SetFromVector(self._last_state)
            else: 
                self.gripping_state = "Waiting for Input"
                self._last_state = q_nominal_arm
                output.SetFromVector(self._last_state)

        else: print("Huston we have a problem")