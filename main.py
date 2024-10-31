from pydrake.all import (
    Simulator,
    StartMeshcat,
    DiagramBuilder,
    SpatialForce,
    AddMultibodyPlantSceneGraph, 
    ModelVisualizer, 
    PackageMap,
    Diagram,
    RigidTransform,
    RotationMatrix,
    SpatialVelocity,
    MultibodyPlant,
    Context,
    AngleAxis,
)
from manipulation import ConfigureParser, FindResource, running_as_notebook
from manipulation.station import AppendDirectives, LoadScenario, MakeHardwareStation

import os
import numpy as np
from typing import BinaryIO, Optional, Union, Tuple
import numpy.typing as npt
import pydot

def export_diagram_as_svg(diagram: Diagram, file: Union[BinaryIO, str]) -> None:
    if type(file) is str:
        file = open(file, "bw")
    graphviz_str = diagram.GetGraphvizString()
    svg_data = pydot.graph_from_dot_data(
        diagram.GetGraphvizString())[0].create_svg()
    file.write(svg_data)

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

### Add objects to scene ###
scenario = AppendDirectives(scenario, filename="objects/added_object_directives.yaml")

station = builder.AddSystem(MakeHardwareStation(
    scenario=scenario,
    meshcat=meshcat,
    # parser_preload_callback=lambda parser: parser.package_map().AddPackageXml("robots/spot_description/package.xml")
    parser_preload_callback=lambda parser: parser.package_map().PopulateFromFolder(os.getcwd())
))

### Get subsystems ###
scene_graph = station.GetSubsystemByName("scene_graph")
plant = station.GetSubsystemByName("plant")

# Add controller = InverseDynamicsController?
# Add cameras?

diagram = builder.Build()
context = diagram.CreateDefaultContext()
diagram.set_name("tidyspot_diagram")
export_diagram_as_svg(diagram, "diagram.svg")

### Simulation setup ###
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyMutableContextFromRoot(context)
plant_context = plant.GetMyMutableContextFromRoot(context)
# controller_context = controller.GetMyMutableContextFromRoot(context)

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




