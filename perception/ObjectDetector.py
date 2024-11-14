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

import os
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
torchvision.disable_beta_transforms_warning()

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


class ObjectDetector(LeafSystem):
    def __init__(self, station: Diagram, camera_names: List[str], image_size: Tuple[int, int], use_groundedsam: bool, groundedsam_path: str = os.path.join(os.getcwd(), "Grounded-Segment-Anything")):
        LeafSystem.__init__(self)

        if use_groundedsam:
            self._setup_groundedsam(groundedsam_path)

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

        # Declare point cloud input ports (Maybe not needed)
        # self._pcd_inputs_indexes = {
        #     camera_name: self.DeclareAbstractInputPort(
        #         f"{camera_name}.point_cloud",
        #         AbstractValue.Make(PointCloud(0))
        #     ).get_index()
        #     for camera_name in camera_names
        # }

        # Internal states

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

        segmentation_mask = self.run_groundedsam(rgb_image)
        camera_name = "frontleft" # Temp

        output.set_value({
            "segmentation_mask": segmentation_mask,  # TODO: Implement this
            "camera_name": camera_name
        })


    def test_segmentation_frontleft(self, object_detector_context: Context):
        rgba_image = self.get_color_image("frontleft", object_detector_context).data
        rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
        mask = self.run_groundedsam(rgb_image)
        print("Frontleft segmentation test complete")
        print(mask.shape)

    def _setup_groundedsam(self, groundedsam_path):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = groundedsam_path + "/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT_PATH = groundedsam_path + "/groundingdino_swint_ogc.pth"

        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = groundedsam_path + "/sam_vit_h_4b8939.pth"

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def run_groundedsam(self, rgb_image):
        # Predict classes and hyper-param for GroundingDINO
        # SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
        CLASSES = [
        "chips can",
        "master chef can",
        "cracker box",
        "sugar box",
        "tomato soup can",
        "mustard bottle",
        "tuna fish can",
        "pudding box",
        "gelatin box",
        "potted meat can",
        "banana",
        "strawberry",
        "apple",
        "lemon",
        "peach",
        "pear",
        "orange",
        "plum",
        "pitcher base",
        "bleach cleanser",
        "windex bottle",
        "wine glass",
        "bowl",
        "mug",
        "sponge",
        "skillet",
        "skillet lid",
        "plate",
        "fork",
        "spoon",
        "knife",
        "spatula",
        "power drill",
        "wood block",
        "scissors",
        "padlock",
        "key",
        "large marker",
        "small marker",
        "adjustable wrench",
        "phillips screwdriver",
        "flat screwdriver",
        "plastic bolt",
        "plastic nut",
        "hammer",
        "small clamp",
        "medium clamp",
        "large clamp",
        "extra large clamp",
        "mini soccer ball",
        "softball",
        "baseball",
        "tennis ball",
        "racquetball",
        "golf ball",
        "chain",
        "foam brick",
        "dice",
        "marbles",
        "cups",
        "colored wood blocks",
        "toy airplane",
        "lego duplo",
        "timer",
        "rubiks cube"
        ]

        BOX_THRESHOLD = 0.3
        TEXT_THRESHOLD = 0.3
        NMS_THRESHOLD = 0.9

        # Detect objects using GroundingDINO
        detections = self.grounding_dino_model.predict_with_classes(
            image=rgb_image,  # Use RGB image here
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # Annotate image with detections (convert RGB to BGR for OpenCV if needed)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections
        ]

        print("Labels before NMS:", labels)

        annotated_frame = box_annotator.annotate(scene=bgr_image.copy(), detections=detections, labels=labels)

        # Save the annotated GroundingDINO image
        cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)

        # NMS post-process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        print("detections.class_id:", detections.class_id)

        print(f"After NMS: {len(detections.xyxy)} boxes")

        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
            return np.array(result_masks)

        # Convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=rgb_image,  # Use RGB image here
            xyxy=detections.xyxy
        )

        # Annotate image with detections and masks
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{CLASSES[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]

        # The MaskAnnotator expects an RGB image
        annotated_image = mask_annotator.annotate(scene=rgb_image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Convert the annotated image back to BGR before saving
        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("grounded_sam_annotated_image.jpg", bgr_annotated_image)
        
        return detections.mask[0]

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