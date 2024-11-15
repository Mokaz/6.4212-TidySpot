import os
import torch
import cv2
import numpy as np
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import torchvision
torchvision.disable_beta_transforms_warning()

import warnings
warnings.filterwarnings("ignore", message="annotate is deprecated:*")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*")

class GroundedSAM:
    def __init__(self, groundedsam_path: str):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_groundedsam(groundedsam_path)
        print("GroundedSAM initialized successfully.")

    def _setup_groundedsam(self, groundedsam_path: str):
        # GroundingDINO config and checkpoint
        GROUNDING_DINO_CONFIG_PATH = os.path.join(groundedsam_path, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(groundedsam_path, "groundingdino_swint_ogc.pth")

        # Segment-Anything checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = os.path.join(groundedsam_path, "sam_vit_h_4b8939.pth")

        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(sam)

    def detect_and_segment_objects(self, rgb_image: np.ndarray):
        CLASSES = [
            "chips can", "master chef can", "cracker box", "sugar box", "tomato soup can", "mustard bottle",
            "tuna fish can", "pudding box", "gelatin box", "potted meat can", "banana", "strawberry", "apple",
            "lemon", "peach", "pear", "orange", "plum", "pitcher base", "bleach cleanser", "windex bottle",
            "wine glass", "bowl", "mug", "sponge", "skillet", "skillet lid", "plate", "fork", "spoon", "knife",
            "spatula", "power drill", "wood block", "scissors", "padlock", "key", "large marker", "small marker",
            "adjustable wrench", "phillips screwdriver", "flat screwdriver", "plastic bolt", "plastic nut", "hammer",
            "small clamp", "medium clamp", "large clamp", "extra large clamp", "mini soccer ball", "softball",
            "baseball", "tennis ball", "racquetball", "golf ball", "chain", "foam brick", "dice", "marbles", "cups",
            "colored wood blocks", "toy airplane", "lego duplo", "timer", "rubiks cube"
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
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # print(f"After NMS: {len(detections.xyxy)} boxes")

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