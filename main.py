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
scenario = AppendDirectives(scenario, filename="objects/added_object_directives.yaml")

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




