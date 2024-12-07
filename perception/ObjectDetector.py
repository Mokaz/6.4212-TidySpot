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
    RigidTransform,
)
from typing import List, Tuple
import matplotlib.pyplot as plt
import torchvision
torchvision.disable_beta_transforms_warning()


import os
import cv2
import numpy as np

from TidySpotFSM import SpotState

class ObjectDetector(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], image_size: Tuple[int, int],
                 use_groundedsam: bool, groundedsam_path: str = os.path.join(os.getcwd(), "third_party/Grounded-Segment-Anything"),
                 device: str = "cpu"):
        LeafSystem.__init__(self)

        if use_groundedsam:
            from perception.GroundedSAM import GroundedSAM
            self.perceptor = GroundedSAM(groundedsam_path, device=device)
        else:
            # if we aren't using GroundedSAM, then we're using groundtruth
            from perception.GroundTruthSensor import GroundTruthSensor
            self.perceptor = GroundTruthSensor(station)

        self._camera_names = camera_names

        # Get the cameras (RgbdSensor) from the HardwareStation
        self._cameras = {
                camera_names: station.GetSubsystemByName(f"rgbd_sensor_{camera_names}")
                for camera_names in camera_names
            }

        # Input ports
        self.fsm_state_input = self.DeclareAbstractInputPort("fsm_state", AbstractValue.Make(SpotState.IDLE))

        self._camera_inputs_indexes = {
            camera_name: {
                image_type: self.DeclareAbstractInputPort(
                    f"{camera_name}.{image_type}",
                    AbstractValue.Make(image_class(image_size[0], image_size[1]))
                ).get_index()
                for image_type, image_class in {
                    'rgb_image': ImageRgba8U,
                    'depth_image': ImageDepth32F,
                    'label_image': ImageLabel16I,
                }.items()
            }
            for camera_name in camera_names
        }

        # Output ports
        self.DeclareAbstractOutputPort(
            "object_detection_segmentations",
            lambda: AbstractValue.Make(dict()),
            self.SegmentAllCameras
        )

        self.DeclareAbstractOutputPort(
            "grasping_object_segmentation",
            lambda: AbstractValue.Make(dict()),
            self.SegmentFrontCameras
        )

    def SegmentAllCameras(self, context: Context, output):
        # print("ObjectDetector: SegmentAllCameras")
        fsm_state = self.fsm_state_input.Eval(context)
        if fsm_state == SpotState.GRASP_OBJECT:
            output.set_value({})
            return

        segmentation_mask_dict = {}
        for camera_name in self._camera_names:
            if camera_name == "back":  # Skip back camera because of obscured view
                continue
            rgb_image = cv2.cvtColor(self.get_color_image(camera_name, context).data, cv2.COLOR_RGBA2RGB)
            label_image = self.get_label_image(camera_name, context).data
            mask, confidence = self.perceptor.detect_and_segment_objects(rgb_image, camera_name, label_image)
            if mask is not None:
                segmentation_mask_dict[camera_name] = mask
        if segmentation_mask_dict:
            print(f"Detected {len(segmentation_mask_dict)} objects! Sending masks to PointCloudCropper")
        else: 
            print("No objects detected")
        output.set_value(segmentation_mask_dict)
        # print("ObjectDetector: SegmentAllCameras complete)

    def SegmentFrontCameras(self, context: Context, output): # For grasp selection
        # print("ObjectDetector: SegmentFrontCameras")
        segmentation_mask_dict = {}

        frontleft_rgb_image = cv2.cvtColor(self.get_color_image("frontleft", context).data, cv2.COLOR_RGBA2RGB)
        frontright_rgb_image = cv2.cvtColor(self.get_color_image("frontright", context).data, cv2.COLOR_RGBA2RGB)
        frontleft_label_image = self.get_label_image("frontleft", context).data
        frontright_label_image = self.get_label_image("frontright", context).data
        front_left_mask, front_left_confidence = self.perceptor.detect_and_segment_objects(frontleft_rgb_image, "frontleft", frontleft_label_image)
        front_right_mask, front_right_confidence = self.perceptor.detect_and_segment_objects(frontright_rgb_image, "frontright", frontright_label_image)

        masks_confidences = [
            (front_left_mask, front_left_confidence, "frontleft"),
            (front_right_mask, front_right_confidence, "frontright")
        ]

        # Filter out None masks
        valid_masks = [(mask, confidence, name) for mask, confidence, name in masks_confidences if mask is not None]

        if valid_masks:
            # Select the mask with the highest confidence
            # print("Detected object! Sending mask to PointCloudCropper")
            segmentation_mask, _, camera_name = max(valid_masks, key=lambda x: x[1])
            segmentation_mask_dict[camera_name] = segmentation_mask
        
        # if segmentation_mask_dict:
        #     print("Object to grasp detected! Sending mask to PointCloudCropper")
        # else:
        #     print("No objects detected")

        output.set_value(segmentation_mask_dict)

        # print("ObjectDetector: GetClosestObjectSegmentation complete")

    def test_segmentation(self, object_detector_context: Context, camera_name):
        rgb_image = cv2.cvtColor(self.get_color_image(camera_name, object_detector_context).data, cv2.COLOR_RGBA2RGB)
        mask, confidence = self.perceptor.detect_and_segment_objects(rgb_image, camera_name)
        if mask is None:
            print(camera_name, " segmentation failed, no object detections found")
        else:
            print("Mask shape:", mask.shape)
            print(camera_name, " segmentation test complete")

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