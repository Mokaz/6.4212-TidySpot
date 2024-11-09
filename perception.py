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
)
from typing import List, Tuple
import matplotlib.pyplot as plt

class CameraHubSystem(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], image_size: Tuple[int, int]):
        LeafSystem.__init__(self)

        self._camera_names = camera_names

        # Get the cameras (RgbdSensor) from the HardwareStation
        self._cameras = {
                camera_names: station.GetSubsystemByName(f"rgbd_sensor_{camera_names}")
                for camera_names in camera_names
            }
        
        # camera_info_frontleft = self._cameras["frontleft"].default_depth_render_camera().core().intrinsics() # EXAMPLE get camera intrinsics

        # Declare input ports for the CameraHubSystem
        self._camera_color_inputs = {
            camera_name: self.DeclareAbstractInputPort(
                f"{camera_name}.rgb_image",
                AbstractValue.Make(ImageRgba8U(image_size[0], image_size[1]))
            )
            for camera_name in camera_names
        }

        self._camera_depth_inputs = {
            camera_name: self.DeclareAbstractInputPort(
                f"{camera_name}.depth_image",
                AbstractValue.Make(ImageDepth32F(image_size[0], image_size[1]))
            )
            for camera_name in camera_names
        }

        self._camera_label_inputs = {
            camera_name: self.DeclareAbstractInputPort(
                f"{camera_name}.label_image",
                AbstractValue.Make(ImageLabel16I(image_size[0], image_size[1]))
            )
            for camera_name in camera_names
        }

    def connect_cameras(self, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            color_input = self._camera_color_inputs[camera_name]
            depth_input = self._camera_depth_inputs[camera_name]
            label_input = self._camera_label_inputs[camera_name]

            builder.Connect(station.GetOutputPort(f"{camera_name}.rgb_image"), color_input)
            builder.Connect(station.GetOutputPort(f"{camera_name}.depth_image"), depth_input)
            builder.Connect(station.GetOutputPort(f"{camera_name}.label_image"), label_input)

    def get_color_image(self, camera_name: str, camera_hub_context: Context):
        return self._camera_color_inputs[camera_name].Eval(camera_hub_context)

    def get_depth_image(self, camera_name: str, camera_hub_context: Context):
        return self._camera_depth_inputs[camera_name].Eval(camera_hub_context)

    def get_label_image(self, camera_name: str, camera_hub_context: Context):
        return self._camera_label_inputs[camera_name].Eval(camera_hub_context)

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