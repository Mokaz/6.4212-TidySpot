import time

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
        self, plant: MultibodyPlant, enabled: bool = True, use_teleop: bool = True
    ):
        super().__init__()

        self._plant = plant
        self.DeclareAbstractInputPort(
            "desired_pose", AbstractValue.Make(RigidTransform())
        )
        # FIXME: allow it to be 10 for now becuase I havn't figured out how to get rid
        # of extra joints in the slider
        self.DeclareVectorInputPort("base_position", 10 if use_teleop else 3)
        self.DeclareVectorOutputPort("desired_spot_arm_position", 7, self._solve_ik)
        self._last_solve = time.time()
        self._last_state = q_nominal_arm

        self._enabled = enabled

    def get_desired_pose_input_port(self):
        return self.GetInputPort("desired_pose")

    def get_base_position_input_port(self):
        return self.GetInputPort("base_position")

    def _solve_ik(self, context: Context, output: BasicVector):
        # because IK is a time-consuming process, we only want to do it once per second
        # in actual application, we shouldn't be solving IK in real time...
        # we should consider using RRT or something similar to plan the motion
        if not self._enabled or time.time() - self._last_solve < 1:
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
                target_frame_name="arm_link_fngr",
                fix_base=True,
                max_iter=1,
            )
            self._last_state = q[3:10]

        except AssertionError:
            self._last_state = q_nominal_arm
        finally:
            output.SetFromVector(self._last_state)
