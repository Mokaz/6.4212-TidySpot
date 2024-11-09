from pydrake.all import (
    Simulator,
    StartMeshcat,
    DiagramBuilder,
)
from manipulation import FindResource, running_as_notebook

from manipulation.station import (
    AddPointClouds,
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
)

from utils import export_diagram_as_svg, DrakeWarningFilter
from perception import CameraHubSystem

import os
import logging
import numpy as np
from matplotlib import pyplot as plt

### Filter out Drake warnings ###
logger = logging.getLogger('drake')
logger.addFilter(DrakeWarningFilter())

### Start the visualizer ###
meshcat = StartMeshcat()
meshcat.AddButton("Stop meshcat")

### Diagram setup ###
builder = DiagramBuilder()
scenario = LoadScenario(
    filename=FindResource(
        "models/spot/spot_with_arm_and_floating_base_actuators.scenario.yaml"
    )
)

camera_names = []
image_size = None

# Enable depth images for all cameras and collect camera names (BUG: False by default?)
for camera_name, camera_config in scenario.cameras.items():
    camera_config.depth = True
    camera_names.append(camera_name)
    
    if image_size is None:
        image_size = (camera_config.width, camera_config.height)

### Add objects to scene ###
from typing import Tuple
import random

def clutter_gen(num_items:int, spawn_area: Tuple[float, float], scene_file: str, goal_file: str):
    ycb = ["003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf", 
        "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"]
    ycb_bases = ["base_link_cracker", "base_link_sugar", "base_link_soup", 
        "base_link_mustard", "base_link_gelatin", "base_link_meat"]
    
    # Step 0: Check if we are in right directory
    current_directory = os.getcwd()
    if not current_directory.endswith("6.4212_TidySpot"):
        scene_path = os.path.join("6.4212_TidySpot", scene_file)
        goal_path = os.path.join("6.4212_TidySpot", goal_file)
    else: 
        scene_path = scene_file
        goal_path = goal_file

    # Step 1: Read the original YAML file
    try:
        with open(scene_path, 'r') as file:
            yaml_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file {scene_path} was not found.")
        return

    # Step 2: Add the area tuple to the YAML content
    rng = random.Random()  # Initialize a random generator
    for i in range(num_items):
        object_num = rng.randint(0, len(ycb)-1)
        x_pos = rng.uniform(-spawn_area[0],spawn_area[0])
        y_pos = rng.uniform(-spawn_area[1],spawn_area[1])
        yaml_content += f"""
- add_model:
    name: thing{i}
    file: package://manipulation/hydro/{ycb[object_num]}
    default_free_body_pose:
        {ycb_bases[object_num]}:
            translation: [{x_pos}, {y_pos}, 2]"""

    # Step 3: Write the modified content to the goal file
    with open(goal_path, 'w') as file:
        file.write(yaml_content)
    print(f"File saved to {goal_path}")

    return goal_file

clutter_file = clutter_gen(4, (3.0, 3.0), "objects/added_object_directives.yaml", "objects/modified_object_directives.yaml")
scenario = AppendDirectives(scenario, filename=clutter_file)

station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,
    parser_preload_callback=lambda parser: parser.package_map().PopulateFromFolder(os.getcwd())
))

### Add pointclouds ###
to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder) # Add meshcat to parameter list to visualize PCDs (WARNING: loads but crashes after a while!)

### Get subsystems ###
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

# Instantiate CameraHubSystem with station and camera_names
camera_hub = builder.AddSystem(CameraHubSystem(station, camera_names, image_size))
camera_hub.set_name("camera_hub")
camera_hub.connect_cameras(station, builder)

# Add controller = InverseDynamicsController?

diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("tidyspot_diagram")
export_diagram_as_svg(diagram, "diagram.svg")

### Simulation setup ###
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

### Get subsystem contexts ###
station_context = station.GetMyMutableContextFromRoot(context)
plant_context = plant.GetMyMutableContextFromRoot(context)
camera_hub_context = camera_hub.GetMyMutableContextFromRoot(context)
# controller_context = controller.GetMyMutableContextFromRoot(context)

################
### TESTZONE ###
################

# Display all camera images 
# camera_hub.display_all_camera_images(camera_hub_context) 

# Display single image
# color_image = camera_hub.get_color_image("frontleft", camera_hub_context) # Get color image from frontleft camera
# plt.imshow(color_image)
# plt.show()

################

### Set initial Spot state ###
x0 = station.GetOutputPort("spot.state_estimated").Eval(station_context)
station.GetInputPort("spot.desired_state").FixValue(station_context, x0)

### Run simulation ###
simulator.set_target_realtime_rate(1)
simulator.set_publish_every_time_step(True)

meshcat.StartRecording()
simulator.AdvanceTo(2.0)
meshcat.PublishRecording()

# Keep meshcat alive
while not meshcat.GetButtonClicks("Stop meshcat"):
    pass
