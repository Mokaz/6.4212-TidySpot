from typing import Optional

from pydrake.all import (
    Diagram,
    DiagramBuilder,
    JointSliders,
    Meshcat,
    MultibodyPlant,
    StateInterpolatorWithDiscreteDerivative,
)

from .position_combiner import PositionCombiner
from .spot_arm_ik_controller import SpotArmIKController


def make_spot_controller(
    spot_plant: MultibodyPlant,
    use_teleop: bool = True,
    meshcat: Optional[Meshcat] = None,
) -> Diagram:
    """
    Create and return a Spot controller that output the state configuration
    (which can be feed into the hardware station to actually move the robot)
    """
    if use_teleop and not meshcat:
        raise ValueError("A meshcat object is required to use teleop")

    builder = DiagramBuilder()

    # Convert positions to desired state
    position_to_state = builder.AddNamedSystem(
        "position_to_state",
        StateInterpolatorWithDiscreteDerivative(
            # TODO: split the control of end effector (the gripper) from the
            # control of the base of Spot
            num_positions=spot_plant.num_positions(),
            time_step=0.05,
            suppress_initial_transient=True,
        ),
    )

    # Combine Spot's arm and base positions
    position_combiner = builder.AddNamedSystem(
        "position_combiner", PositionCombiner(use_teleop)
    )

    builder.Connect(
        position_combiner.get_output_port(), position_to_state.get_input_port()
    )

    if use_teleop:
        # Pose Sliders
        teleop = builder.AddNamedSystem(
            "teleop",
            JointSliders(
                meshcat,
                spot_plant,
                spot_plant.GetPositions(spot_plant.CreateDefaultContext()),
                step=0.05,
                decrement_keycodes=["ArrowLeft", "ArrowDown", "KeyA"] + [""] * 7,
                increment_keycodes=["ArrowRight", "ArrowUp", "KeyD"] + [""] * 7,
            ),
        )
        builder.Connect(
            teleop.get_output_port(), position_combiner.get_base_position_input_port()
        )
        builder.Connect(
            teleop.get_output_port(),
            position_combiner.get_arm_position_input_port(),
        )

    # Export I/O ports
    if not use_teleop:
        builder.ExportInput(
            position_combiner.get_base_position_input_port(), "desired_base_position"
        )
        builder.ExportInput(
            position_combiner.get_arm_position_input_port(), "desired_arm_position"
        )
    builder.ExportOutput(position_to_state.get_output_port(), "desired_state")

    # Finalize the diagram
    spot_controller = builder.Build()
    spot_controller.set_name("spot_controller" + ("_teleoped" if use_teleop else ""))
    return spot_controller


# alias, since we can think of the function as a "constructor"
SpotController = make_spot_controller
