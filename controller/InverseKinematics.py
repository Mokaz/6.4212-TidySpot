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

q_nominal_arm = np.array([0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0])

def solve_ik(
    plant: MultibodyPlant,
    X_WT: RigidTransform,
    # target_frame_name: str = "arm_link_wr1",
    target_frame_name: str = "arm_link_fngr",
    base_position: np.ndarray = np.zeros(3),
    fix_base: bool = True,
    rotation_bound: float = 0.01,
    position_bound: float = 0.01,
    assert_error_on_fail: bool = True,
    q_current=q_nominal_arm,
    pure_vertical_movement: bool = False,
    max_iter: int = 20,
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
        context = plant.CreateDefaultContext()
        ik = InverseKinematics(plant, context)
        q = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram

        world_frame = plant.world_frame()
        target_frame = plant.GetFrameByName(target_frame_name)

        # nominal pose
        q0 = np.zeros(len(q))
        q0[:3] = base_position
        q0[3:10] = q_current

        # Target position and rotation
        p_WT = X_WT.translation()
        R_WT = X_WT.rotation()

        if pure_vertical_movement:
            # Position constraint: Keep x and y fixed at p_WT[0], p_WT[1],
            # allow z to vary only downward by position_bound.
            p_AQ_lower = np.array([p_WT[0], p_WT[1], p_WT[2] - position_bound])
            p_AQ_upper = np.array([p_WT[0], p_WT[1], p_WT[2]])
        else:
            # Position constraint: Allow full 3D movement within position_bound.
            p_AQ_lower = p_WT - position_bound
            p_AQ_upper = p_WT + position_bound

        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=target_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_AQ_lower,
            p_AQ_upper=p_AQ_upper,
        )

        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=R_WT,
            frameBbar=target_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=rotation_bound,
        )

        if fix_base:
            prog.AddConstraint(eq(q[:3], base_position))

        q_start = q0.copy()
        q_start[3:10] = q_current
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q_start)

        result = Solve(ik.prog())
        if result.is_success():
            print("IK successfully solved")
            return result.GetSolution(q)
    if assert_error_on_fail:
        raise AssertionError("IK failed to converge")
    return None
