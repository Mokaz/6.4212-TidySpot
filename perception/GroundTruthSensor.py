import numpy as np
from pydrake.geometry import RenderEngineVtkParams, MakeRenderEngineVtk
from pydrake.geometry import PerceptionProperties, RenderLabel, Role


def assign_label_to_body_for_renderer(plant, scene_graph, body, label_id, renderer_name):
    perception_props = PerceptionProperties()
    perception_props.AddProperty("label", "id", RenderLabel(label_id))
    for geometry_id in plant.GetVisualGeometriesForBody(body):
        scene_graph.AssignRole(
            source_id=plant.get_source_id(),
            geometry_id=geometry_id,
            properties=perception_props,
            role=Role.kPerception,
            renderer=renderer_name,  # Assign to the specific renderer
            skip_property_merge=False,
        )

class GroundTruthSensor:
    #TODO: implement
    def __init__(self, station, camera_names, label_camera_names, image_size, use_grounded_sam, device='cpu'):
        super().__init__()
        self.station = station
        self.camera_names = camera_names
        self.label_camera_names = label_camera_names
        self.image_size = image_size
        self.use_grounded_sam = use_grounded_sam

        # Get plant and scene_graph
        plant = self.station.GetSubsystemByName("plant")
        scene_graph = self.station.GetSubsystemByName("scene_graph")

            # Add renderers
        original_renderer_name = "original_renderer"
        scene_graph.AddRenderer(original_renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

        label_renderer_name = "label_renderer"
        scene_graph.AddRenderer(label_renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

        model_labels = {
            'book1': 1,
            'book2': 2,
            'book3': 3,
            'book4': 4,
            'planar_bin': 5,
            'table': 6,
        }

        for model_name, label_id in model_labels.items():
            model_instance = plant.GetModelInstanceByName(model_name)
            body_indices = plant.GetBodyIndices(model_instance)
            for body_index in body_indices:
                body = plant.get_body(body_index)
                assign_label_to_body_for_renderer(
                    plant, scene_graph, body, label_id, label_renderer_name)

        

        # Declare input ports for color images (original cameras)
        self.color_image_input_ports = {}
        for camera_name in camera_names:
            port = self.DeclareAbstractInputPort(
                f"{camera_name}_rgb_image",
                AbstractValue.Make(ImageRgba8U())
            )
            self.color_image_input_ports[camera_name] = port

        # Declare input ports for label images (label cameras)
        self.label_image_input_ports = {}
        for camera_name in label_camera_names:
            port = self.DeclareAbstractInputPort(
                f"{camera_name}_label_image",
                AbstractValue.Make(ImageLabel16I())
            )
            self.label_image_input_ports[camera_name] = port

        # Rest of your initialization code...

    def connect_cameras(self, station, builder):
        # Connect the color image output ports of the original cameras
        for camera_name in self.camera_names:
            camera_system = station.GetSubsystemByName(camera_name)
            color_image_output_port = camera_system.color_image_output_port()
            builder.Connect(
                color_image_output_port,
                self.get_input_port(f"{camera_name}_rgb_image")
            )

        # Connect the label image output ports of the label cameras
        for camera_name in self.label_camera_names:
            camera_system = station.GetSubsystemByName(camera_name)
            label_image_output_port = camera_system.label_image_output_port()
            builder.Connect(
                label_image_output_port,
                self.get_input_port(f"{camera_name}_label_image")
            )
    def detect_and_segment_objects(self, rgb_image: np.ndarray, camera_name: str):
        pass