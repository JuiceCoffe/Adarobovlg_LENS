"""SAM3 segmentation and optional remote visual grounding."""

import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from openai import OpenAI
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor


def normalize_bbox(box, width, height):
    """Normalize pixel-space bounding-box coordinates to the [0, 1] range.

    Args:
        box: Bounding box formatted as [xmin, ymin, xmax, ymax].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Normalized coordinates [xmin_norm, ymin_norm, xmax_norm, ymax_norm].
    """
    xmin, ymin, xmax, ymax = box
    xmin_norm = xmin / width
    ymin_norm = ymin / height
    xmax_norm = xmax / width
    ymax_norm = ymax / height
    return [xmin_norm, ymin_norm, xmax_norm, ymax_norm]


def recover_bbox(norm_box, width, height):
    """Convert normalized bounding-box coordinates back to pixel coordinates.

    Args:
        norm_box: Normalized coordinates [xmin_norm, ymin_norm, xmax_norm, ymax_norm].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Pixel coordinates [xmin, ymin, xmax, ymax].
    """
    xmin_norm, ymin_norm, xmax_norm, ymax_norm = norm_box
    xmin = int(xmin_norm * width)
    ymin = int(ymin_norm * height)
    xmax = int(xmax_norm * width)
    ymax = int(ymax_norm * height)
    return [xmin, ymin, xmax, ymax]


def prepare_rgb_images(images):
    """Normalize processor input to contiguous uint8 RGB images in NxHxWx3."""
    images_np = images if isinstance(images, np.ndarray) else np.asarray(images)
    assert (
        images_np.ndim == 4 and images_np.shape[-1] == 3
    ), "Input images should be in NxHxWx3 format"

    if images_np.dtype != np.uint8:
        images_np = images_np.astype(np.float32)
        if images_np.size > 0 and images_np.max() <= 1.0:
            images_np = images_np * 255.0
        images_np = np.clip(images_np, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(images_np)


def _to_numpy(value):
    """Detach tensors to CPU NumPy arrays, or convert existing array-like values."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_instance_mask(result, image_shape):
    """Union nonempty instance masks occupying at most 60% of the image; return mask and boxes."""
    masks = result.get("masks", [])
    if len(masks) == 0:
        return None, []

    boxes = result.get("boxes", None)
    img_area = float(image_shape[0] * image_shape[1])
    selected_masks = []
    selected_boxes = []

    for index in range(len(masks)):
        mask_np = _to_numpy(masks[index]).astype(bool)
        area = float(mask_np.sum())
        area_ratio = area / max(img_area, 1.0)
        if area <= 0 or area_ratio > 0.6:
            continue

        selected_masks.append(mask_np)
        if boxes is not None and len(boxes) > index:
            selected_boxes.append(_to_numpy(boxes[index]).tolist())

    if not selected_masks:
        return None, []

    # Text prompts can refer to several instances, e.g. "fruit" should include apple and banana.
    union_mask = np.logical_or.reduce(selected_masks)
    return union_mask, selected_boxes


class MaskGenerator:
    def __init__(self, confidence_thres=0.7, ckpt_path="checkpoints/sam3", gen_box=True, backend=None):
        """Load local SAM3 processors/models and select remote-box or local-text segmentation."""
        from utils.checkpoints import ROOT, checkpoint_path

        backend = backend or os.environ.get("ADAROBOVLG_MASK_BACKEND", "sam3").lower()
        if backend == "lens":
            from .lens_mask_generator import LensMaskGenerator

            self.backend = "lens"
            self.generate_mask = LensMaskGenerator().generate_mask
            self.generate_mask_with_2d_points = None
            return
        if backend != "sam3":
            raise ValueError(f"Unknown mask backend: {backend}. Expected 'sam3' or 'lens'.")
        self.backend = "sam3"

        relative = (
            "sam3" if ckpt_path == "checkpoints/sam3" else ROOT / Path(ckpt_path).expanduser()
        )
        ckpt_path = str(checkpoint_path(relative, "ADAROBOVLG_SAM3_PATH", directory=True))
        self.device = Accelerator().device
        self.confidence_thres = confidence_thres
        self.gen_box = gen_box
        self.ckpt_path = ckpt_path
        self.point_model = None
        self.point_processor = None

        if gen_box:
            self.model = Sam3TrackerModel.from_pretrained(ckpt_path).to(self.device)
            self.processor = Sam3TrackerProcessor.from_pretrained(ckpt_path)
            self.od = ObjectDetector()
            self.generate_mask = self.generate_mask_with_box
        else:
            self.model = Sam3Model.from_pretrained(ckpt_path).to(self.device)
            self.processor = Sam3Processor.from_pretrained(ckpt_path)
            self.generate_mask = self.generate_mask_without_box

    def generate_mask_with_box(self, images, prompt):
        """Generate masks for RGB images shaped (N, H, W, 3) using box prompts."""
        # box = [x, y, w, h] in pixel coordinates
        # preprocess
        images_np = prepare_rgb_images(images)

        # boxes = self.od.object_detect(prompt, images_np)
        boxes = [self.od.object_detect(prompt, img[None, ...])[0] for img in images_np]  # [[]]
        boxes = [
            (
                recover_bbox(
                    normalize_bbox(box, 1000, 1000), images_np.shape[2], images_np.shape[1]
                )
                if len(box) > 0
                else []
            )
            for box in boxes
        ]  # list of N boxes, each box is [x1,y1,x2,y2] or []
        valid_idxs = [i for i, box in enumerate(boxes) if len(box) > 0]
        valid_idxs = np.array(valid_idxs)
        if len(valid_idxs) == 0:
            print("No valid box detected for the prompt.")
            self.masks = np.zeros(images_np.shape[:3], dtype=bool)
            return self.masks, [[]] * len(images_np)

        valid_images = images_np[valid_idxs]  # NxHxWx3
        valid_boxes = [boxes[i] for i in valid_idxs]  # [[]]
        valid_boxes = np.array(valid_boxes)[:, None, :]  # Nx1x4
        valid_labels = np.ones(valid_boxes.shape[0], dtype=int).reshape(-1, 1, 1)  # Nx1x1

        inputs = self.processor(
            images=valid_images,
            input_boxes=valid_boxes,
            input_labels=valid_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        valid_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], mask_threshold=self.confidence_thres
        )  # list of N masks, each mask is 1x3xHxW
        valid_masks = torch.stack(valid_masks, dim=0)[:, 0, -1, :, :].cpu().numpy()  # NxHxW
        masks = np.zeros(images_np.shape[:3], dtype=bool)
        masks[valid_idxs] = valid_masks
        return masks, boxes


    def generate_mask_without_box(self, images, prompt):
        """Generate masks for RGB images shaped (N, H, W, 3) using text prompts."""
        images_np = prepare_rgb_images(images)

        # repeat prompt
        prompts = [prompt] * images_np.shape[0]
        # Segment each image with the repeated text prompt.
        inputs = self.processor(images=images_np.copy(), text=prompts, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert raw predictions into instance masks.
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_thres,
            mask_threshold=self.confidence_thres,
            target_sizes=inputs.get("original_sizes").tolist(),
        )

        # Consolidate selected instances into one mask per view.
        masks = np.zeros(images_np.shape[:3], dtype=bool)
        boxes = [[]] * len(images_np)

        for i, result in enumerate(results):
            selected_mask, selected_box = select_instance_mask(result, images_np.shape[1:3])
            if selected_mask is not None:
                masks[i] = selected_mask
                boxes[i] = selected_box

        return masks, boxes

    def generate_mask_with_2d_points(self, images, points_2d):
        """Generate masks for RGB images using per-view 2D point prompts."""
        # points in pixel coordinates
        # preprocess
        images_np = prepare_rgb_images(images)

        def _valid_points(points):
            """Keep only two-coordinate point prompts, treating missing points as an empty list."""
            if points is None:
                return []
            if isinstance(points, np.ndarray):
                points = points.tolist()
            return [point for point in points if len(point) == 2]

        points_2d = [_valid_points(points) for points in points_2d]
        valid_indices = [i for i, points in enumerate(points_2d) if len(points) > 0]
        masks = np.zeros(images_np.shape[:3], dtype=bool)
        if len(valid_indices) == 0:
            return masks

        if self.gen_box:
            point_model, point_processor = self.model, self.processor
        else:
            if self.point_model is None:
                self.point_model = Sam3TrackerModel.from_pretrained(self.ckpt_path).to(self.device)
                self.point_processor = Sam3TrackerProcessor.from_pretrained(self.ckpt_path)
            point_model, point_processor = self.point_model, self.point_processor

        valid_images = images_np[valid_indices]
        valid_points = [points_2d[i] for i in valid_indices]
        input_points = [[points] for points in valid_points]  # list of N points, each point is 1x2
        input_labels = [[[1] * len(points)] for points in valid_points]  # Nx1xK

        inputs = point_processor(
            images=valid_images.copy(),
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = point_model(**inputs)
        valid_masks = point_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], mask_threshold=self.confidence_thres
        )  # list of N masks, each mask is 1x3xHxW
        valid_masks = torch.stack(valid_masks, dim=0)[:, 0, -1, :, :].cpu().numpy()  # NxHxW
        if len(valid_masks) != len(valid_indices):
            print(
                f"Warning: SAM returned {len(valid_masks)} masks for {len(valid_indices)} valid point prompts."
            )
        for mask_idx, view_idx in enumerate(valid_indices[: len(valid_masks)]):
            masks[view_idx] = valid_masks[mask_idx]

        # Reject large masks that likely represent the background.
        mask_areas = masks.sum(axis=(1, 2))  # N
        img_areas = images_np.shape[1] * images_np.shape[2]
        area_ratios = mask_areas / img_areas
        masks[area_ratios > 0.1] = 0
        for idx, area_ratio in enumerate(area_ratios):
            if area_ratio > 0.1:
                print(
                    f"Mask {idx} has a large area ratio of {area_ratio:.2f}, likely a background mask, setting it to zero."
                )

        return masks




def encode_image_opencv(image):
    """Base64-encode file bytes or an OpenCV-order array without changing channels."""
    if isinstance(image, str) and os.path.isfile(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("utf-8")


class ObjectDetector:
    def __init__(self, model_name="qwen3.5-plus"):
        """Configure the remote bounding-box detector from environment settings."""
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError(
                "Set DASHSCOPE_API_KEY for box-prompt segmentation, or use --text-prompt."
            )
        self.client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=os.environ.get(
                "ADAROBOVLG_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
        )
        self.model_name = os.environ.get("ADAROBOVLG_VLM_MODEL", model_name)
        self.messages = [
            {
                "role": "system",
                "content": """
                You are an expert in 2D object detection. Given a language instruction,
                identify the requested object or object part in one or more images and
                return its bounding box.

                Requirements:
                1. Identify the object or part that best matches the instruction.
                2. Instructions may describe examples such as:
                - "the red apple"
                - "the book on the table"
                - "the red power drill"
                - "the drill head"
                - "the drill handle"
                3. Return an accurate bounding box as [xmin, ymin, xmax, ymax].
                4. Express every coordinate in the normalized 0-1000 range.
                5. When multiple images are provided, locate the same target in every image.
                6. Return exactly one bounding-box entry per image, in input order.
                7. The description may be approximate; use matching color, shape, and
                   function to identify the most likely target.

                Output format:
                Return a strict JSON array using one of these forms:
                [{"bbox": [100, 150, 200, 250]}],
                [{"bbox": [50, 80, 120, 160]}, {"bbox": [300, 400, 350, 450]}],
                [{"bbox": [60, 90, 110, 140]}, {"bbox": []}],

                Constraints:
                - Return only the JSON array, without Markdown or explanatory text.
                - If several objects match, return only the one most relevant to the instruction.
                - The number and order of entries must match the input images.
                - Use an empty bbox list when the target is not visible in an image.
                """,
            },
        ]

    def object_detect(self, prompt, images):
        """Request one normalized [xmin, ymin, xmax, ymax] box per image; missing detections use []."""
        encoded_images = [encode_image_opencv(image) for image in images]
        self.messages = self.messages[:1]  # Retain only the system message between requests.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                        Identify the object or object part described by the following
                        instruction and return its bounding-box coordinates:
                        "{prompt}"
                        Return JSON in this format:
                        [{"bbox": [xmin, ymin, xmax, ymax]}]
                        """.replace(
                            "prompt", prompt
                        ),
                    },
                ],
            }
        )
        for img in encoded_images:
            self.messages[-1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                },
            )

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            extra_body={"enable_thinking": False},  # Disable extended reasoning to reduce latency.
            response_format={"type": "json_object"},
        )
        response: str = completion.choices[0].message.content
        data = json.loads(response)
        result = []
        if isinstance(data, list):
            num_box = len(data)
            num_image = len(encoded_images)
            if num_box > num_image:
                data = data[:num_image]
            elif num_box < num_image:
                data.extend([{"bbox": []}] * (num_image - num_box))
            for item in data:
                if (
                    isinstance(item, dict)
                    and "bbox" in item
                    and isinstance(item["bbox"], list)
                    and len(item["bbox"]) == 4
                ):
                    result.append(item["bbox"])
                elif (
                    isinstance(item, list)
                    and len(item) > 0
                    and isinstance(item[0], dict)
                    and "bbox" in item[0]
                    and isinstance(item[0]["bbox"], list)
                    and len(item[0]["bbox"]) == 4
                ):
                    result.append(item[0]["bbox"])
                else:
                    print("Invalid bbox format in item:", item)
                    result.append([])
            return result
        else:
            print("Invalid response format:", response)
            return [[]] * len(encoded_images)
