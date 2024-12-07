import numpy as np
from pydrake.geometry import RenderEngineVtkParams, MakeRenderEngineVtk
from pydrake.geometry import PerceptionProperties, RenderLabel, Role

from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    LeafSystem,
    Context,
    ImageRgba8U,
    ImageDepth32F,
    ImageLabel16I,
    Diagram,
    PointCloud,
    DepthImageToPointCloud,
    RigidTransform,
)
import cv2
from skimage import color
from PIL import Image

from pydrake.geometry import RoleAssign

class GroundTruthSensor:
    #TODO: implement
    def __init__(self, station):
        self.station = station

        # Get plant and scene_graph
        plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")
        # Get the inspector
        inspector = self.scene_graph.model_inspector()

        # Retrieve all geometry IDs
        all_geometry_ids = inspector.GetAllGeometryIds()

        # Initialize the label-to-object mapping
        label_to_object = {}

        # Iterate over all geometries
        for geometry_id in all_geometry_ids:
            # Get the perception properties of the geometry
            perception_props = inspector.GetPerceptionProperties(geometry_id)
            if perception_props is None:
                continue
            if perception_props.HasProperty("label", "id"):
                render_label = perception_props.GetProperty("label", "id")
                # Map the render label to an object identifier
                object_name = inspector.GetName(geometry_id)
                if object_name.startswith("obj_"):
                    label_to_object[object_name] = int(render_label)

        self.models = label_to_object
        # Get the indices of each model
        #model_indices =

    def detect_and_segment_objects(self, rgb_image: np.ndarray, camera_name: str, label_image: np.ndarray):
        """detect and segment

        Args:
            rgb_image (np.ndarray): _description_
            camera_name (str): _description_
            label_image (np.ndarray): _description_

        Returns:
            mask: image of bools, where true represents object and false represents not an object
            confidence: float, the confidence value
        """
        # Extract the target indices from the models dictionary
        target_indices = list(self.models.values())

        annotated_image = np.squeeze(color.label2rgb(label_image, bg_label=32766))
        cv2.imwrite("ground_truth_annotated_image.jpg", 255*annotated_image)
        # Create a boolean mask where True if the pixel matches any target index
        mask = np.isin(label_image, target_indices)
        if not np.any(mask):
            return None, None
        return mask, 1
