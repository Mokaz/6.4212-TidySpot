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
from typing import List, Tuple, Mapping
import matplotlib.pyplot as plt

class ObjectDetector(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], image_size: Tuple[int, int]):
        LeafSystem.__init__(self)

        self._camera_names = camera_names

        # Get the cameras (RgbdSensor) from the HardwareStation
        self._cameras = {
                camera_names: station.GetSubsystemByName(f"rgbd_sensor_{camera_names}")
                for camera_names in camera_names
            }

        # Declare input ports for the ObjectDetector
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

        # Declare point cloud input ports (Maybe not needed)
        # self._pcd_inputs_indexes = {
        #     camera_name: self.DeclareAbstractInputPort(
        #         f"{camera_name}.point_cloud",
        #         AbstractValue.Make(PointCloud(0))
        #     ).get_index()
        #     for camera_name in camera_names
        # }

    def connect_cameras(self, station: Diagram, builder: DiagramBuilder):
        for camera_name in self._camera_names:
            for image_type in ['rgb_image', 'depth_image', 'label_image']:
                builder.Connect(
                    station.GetOutputPort(f"{camera_name}.{image_type}"),
                    self.get_input_port(self._camera_inputs_indexes[camera_name][image_type])
                )

    # def connect_point_clouds(self, to_point_cloud: Mapping[str, DepthImageToPointCloud], builder: DiagramBuilder):
    #     for camera_name in self._camera_names:
    #         builder.Connect(
    #             to_point_cloud[camera_name].GetOutputPort(f"point_cloud"), 
    #             self.get_input_port(self._pcd_inputs_indexes[camera_name])
    #         )

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