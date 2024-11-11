import numpy as np
from pydrake.all import (
    Context,
    InverseKinematics,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Solve,
    eq,
)

# nominal joint angles for Spot's arm (for joint centering)
q_nominal_arm = np.array([0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0])
# for base, we can just use whatever the current pose is


def sample_initial_q(plant: MultibodyPlant) -> np.ndarray:
    lower_bound = np.nan_to_num(plant.GetPositionLowerLimits(), neginf=-np.pi)
    upper_bound = np.nan_to_num(plant.GetPositionUpperLimits(), posinf=np.pi)
    lower_bound + (upper_bound - lower_bound) * np.random.rand(len(lower_bound))


def solve_ik(
    plant: MultibodyPlant,
    context: Context,
    X_WT: RigidTransform,
    target_frame_name: str = "arm_link_fngr",
    base_position: np.ndarray = np.zeros(3),
    fix_base: bool = True,
    rotation_bound: float = 0.01,
    position_bound: float = 0.01,
    collision_bound: float = 0.001,
    error_on_fail: bool = True,
    q_current=q_nominal_arm,
    max_iter: int = 10,
):
    """Convert the desired pose for Spot to joint angles, subject to constraints.

    Args:
        plant (MultibodyPlant): The plant that contains the Spot model.
        context (Context): The plant context
        X_WT (RigidTransform): The target pose in the world frame.
        target_frame_name (str, optional): The name of a frame that X_WT should correspond to,
        defaults to "arm_link_fngr" (the upper part of the gripper on Spot's arm).
        fix_base (bool, optional): If True, then the body of Spot will be fixed to the current
        pose. Defaults to True.
        rotation_bound (float, optional): The maximum allowed rotation error in radians.
        position_bound (float, optional): The maximum allowed position error.
        collision_bound (float, optional): The minimum allowed distance between Spot and the other
        objects in the scene.
    """
    for _ in range(max_iter):
        ik = InverseKinematics(plant, context)
        q = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram

        world_frame = plant.world_frame()
        target_frame = plant.GetFrameByName(target_frame_name)

        # nominal pose
        q0 = np.zeros(len(q))
        q0[:3] = base_position
        q0[3:10] = q_nominal_arm

        # Target position and rotation
        p_WT = X_WT.translation()
        R_WT = X_WT.rotation()

        # Constraints
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=target_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_WT - position_bound,
            p_AQ_upper=p_WT + position_bound,
        )
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=R_WT,
            frameBbar=target_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=rotation_bound,
        )
        # # This is currently failing for some reason
        # # collision constraint
        # ik.AddMinimumDistanceLowerBoundConstraint(
        #     collision_bound, influence_distance_offset=0.001
        # )

        if fix_base:
            prog.AddConstraint(eq(q[:3], base_position))

        # Let's get started
        q_start = q0.copy()
        q_start[3:10] = q_current
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q_start)

        result = Solve(ik.prog())
        if result.is_success():
            return result.GetSolution(q)
    if error_on_fail:
        raise AssertionError("IK failed :(")
    return None
