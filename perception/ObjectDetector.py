import numpy as np
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
)
from typing import List, Tuple
import matplotlib.pyplot as plt
import torchvision
torchvision.disable_beta_transforms_warning()

from perception.GroundedSAM import GroundedSAM

import os
import cv2
import numpy as np

class ObjectDetector(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], image_size: Tuple[int, int], use_groundedsam: bool, groundedsam_path: str = os.path.join(os.getcwd(), "Grounded-Segment-Anything")):
        LeafSystem.__init__(self)

        if use_groundedsam:
            self.grounded_sam = GroundedSAM(groundedsam_path)

        self._camera_names = camera_names

        # Get the cameras (RgbdSensor) from the HardwareStation
        self._cameras = {
                camera_names: station.GetSubsystemByName(f"rgbd_sensor_{camera_names}")
                for camera_names in camera_names
            }

        # Input ports
        self._camera_inputs_indexes = {
            camera_name: {
                image_type: self.DeclareAbstractInputPort(
                    f"{camera_name}.{image_type}",
                    AbstractValue.Make(image_class(image_size[0], image_size[1]))
                ).get_index()
                for image_type, image_class in {
                    'rgb_image': ImageRgba8U,
                    'depth_image': ImageDepth32F,
                    'label_image': ImageLabel16I
                }.items()
            }
            for camera_name in camera_names
        }

        # Output ports
        self.DeclareAbstractOutputPort(
            "segmentation_data",
            lambda: AbstractValue.Make({
                "segmentation_mask": np.array([]),
                "camera_name": ""
            }),
            self.GetClosestObjectSegmentation
        )

    def GetClosestObjectSegmentation(self, context: Context, output):
        # run_GroundedSAM(), or fetch object segmentations from internal state # TODO: implement this

        closest_object = np.array([0]) # TODO: Implement this

        rgba_image = self.get_color_image("frontleft", context).data
        rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)

        segmentation_mask = self.grounded_sam.detect_and_segment_objects(rgb_image)
        camera_name = "frontleft" # Temp


        output.set_value({
            "segmentation_mask": segmentation_mask,  # TODO: Implement this
            "camera_name": camera_name
        })

        # print("ObjectDetector: GetClosestObjectSegmentation complete")

    def test_segmentation_frontleft(self, object_detector_context: Context):
        rgba_image = self.get_color_image("frontleft", object_detector_context).data
        rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
        mask = self.grounded_sam.detect_and_segment_objects(rgb_image)
        print("Frontleft segmentation test complete")
        print(mask.shape)

    def connect_cameras(self, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            for image_type in ['rgb_image', 'depth_image', 'label_image']:
                builder.Connect(
                    station.GetOutputPort(f"{camera_name}.{image_type}"),
                    self.get_input_port(self._camera_inputs_indexes[camera_name][image_type])
                )

    def get_image(self, camera_name: str, image_type: str, camera_hub_context: Context):
        return self.get_input_port(self._camera_inputs_indexes[camera_name][image_type]).Eval(camera_hub_context)

    def get_color_image(self, camera_name: str, camera_hub_context: Context):
        return self.get_image(camera_name, 'rgb_image', camera_hub_context)

    def get_depth_image(self, camera_name: str, camera_hub_context: Context):
        return self.get_image(camera_name, 'depth_image', camera_hub_context)

    def get_label_image(self, camera_name: str, camera_hub_context: Context):
        return self.get_image(camera_name, 'label_image', camera_hub_context)

    def display_all_camera_images(self, camera_hub_context: Context):
        fig, axes = plt.subplots(len(self._camera_names), 3, figsize=(15, 5 * len(self._camera_names)))

        for i, camera_name in enumerate(self._camera_names):
            color_img = self.get_color_image(camera_name, camera_hub_context).data
            depth_img = self.get_depth_image(camera_name, camera_hub_context).data
            label_img = self.get_label_image(camera_name, camera_hub_context).data

            # Plot the color image.
            axes[i, 0].imshow(color_img)
            axes[i, 0].set_title(f"{camera_name} Color image")
            axes[i, 0].axis('off')

            # Plot the depth image.
            axes[i, 1].imshow(np.squeeze(depth_img))
            axes[i, 1].set_title(f"{camera_name} Depth image")
            axes[i, 1].axis('off')

            # Plot the label image.
            axes[i, 2].imshow(label_img)
            axes[i, 2].set_title(f"{camera_name} Label image")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()