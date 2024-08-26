import torch
import os
import cv2
import glob
import numpy as np
from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def load_image(image_str: str) -> Image.Image:
    if image_str.startswith("http"):
        image = Image.open(requests.get(image_str, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_str).convert("RGB")
    return image

def get_boxes(results: List[DetectionResult]) -> List[List[float]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_masks(masks: torch.BoolTensor) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int().numpy().astype(np.uint8)
    return list(masks)

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]

    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    return [DetectionResult.from_dict(result) for result in results]

def segment(
    image: Image.Image,
    detection_results: List[DetectionResult],
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, segmenter_id)

    return detections

def save_masks(detections: List[DetectionResult], save_path: str) -> None:
    for idx, detection in enumerate(detections):
        mask = detection.mask
        if mask is not None:
            mask_save_path = save_path
            cv2.imwrite(mask_save_path, mask * 255)  # Save as grayscale image

waterbird_dir = '/home/mila/j/jaewoo.lee/scratch/dataset/waterbirds/waterbird_complete95_forest2water2'
image_list = glob.glob(f'{waterbird_dir}/**/*.jpg', recursive=True)

a = 0
for image_path in image_list:
    if a < 7655:
        a += 1
        continue
    else:
        a += 1
    folder_name = os.path.join('/home/mila/j/jaewoo.lee/projects/text_prompt_sam/waterbird_results_mask/', 
                               os.path.dirname(image_path).split('waterbird_complete95_forest2water2/')[-1])
    file_name = os.path.join('/home/mila/j/jaewoo.lee/projects/text_prompt_sam/waterbird_results_mask/', 
                             image_path.split('waterbird_complete95_forest2water2/')[-1])

    os.makedirs(folder_name, exist_ok=True)
    print('Target: ', file_name)

    labels = ["a bird."]
    threshold = 0.2

    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    print('before seg')
    detections = grounded_segmentation(
        image=image_path,
        labels=labels,
        threshold=threshold,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )
    print('after seg')

    save_masks(detections, file_name)

