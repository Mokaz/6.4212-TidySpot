import argparse
import os

from TidySpot import run_TidySpot


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Runs the TidySpot project and simulation"
    )

    parser.add_argument(
        "--grasp_type",
        type=str,
        choices=["anygrasp", "antipodal"],
        default="antipodal",
        help="Method used to do grasps",
    )
    parser.add_argument(
        "--perception_type",
        type=str,
        choices=["ground_truth", "sam"],
        default="ground_truth",
        help="Method used to do perception",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="objects/added_object_directives.yaml",
        # others:
        # "objects/blank_scene.yaml"
        # "objects/dorm_room_scene.yaml"
        help="Path to scenario yaml file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Which device to load models on, (cpu, cuda, cuda:1)",
    )

    # Parse the arguments
    args = parser.parse_args()
    run_TidySpot(args)

