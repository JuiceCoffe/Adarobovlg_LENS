"""LENS based natural-language to mask adapter.

This module keeps LENS isolated from the existing SAM3 implementation.  It
loads a LENS checkpoint once and exposes the same ``(masks, boxes)`` return
value used by :class:`MaskGenerator`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import smart_resize
from torchvision.transforms.functional import resize as resize_api
from transformers import AutoProcessor, logging


class LensMaskGenerator:
    """Run the released LENS Qwen2.5-VL + SAM2 checkpoint on RGB images."""

    def __init__(self, checkpoint=None, device=None, dtype=None):
        repo = Path(os.environ.get("ADAROBOVLG_LENS_REPO", "lens_src")).expanduser()
        if not repo.is_absolute():
            repo = Path(__file__).resolve().parents[1] / repo
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from src.open_r1.constants import question_template_registry, system_prompt_registry
        from src.open_r1.trainer.samr1 import SAMR1ForConditionalGeneration_qwen2p5
        from src.open_r1.utils import parse_float_sequence_within

        self._parse_box = parse_float_sequence_within
        self._question_template = question_template_registry["pr1_grounding"]
        self._system_template = system_prompt_registry["default"]
        checkpoint = checkpoint or os.environ.get(
            "ADAROBOVLG_LENS_PATH", str(repo / "pretrained" / "qwen2p5_refcoco")
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        logging.set_verbosity_error()
        # Match the released LENS demo, which explicitly uses SDPA.
        attention_impl = "sdpa" if self.device.type == "cuda" else "eager"
        self.model = SAMR1ForConditionalGeneration_qwen2p5.from_pretrained(
            checkpoint,
            torch_dtype=self.dtype,
            attn_implementation=attention_impl,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def _prepare(self, image, instruction):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.size and image.max() <= 1.0:
                image *= 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image, mode="RGB")
        width, height = image.size
        # Match lens_src/demo.py: larger budgets change Qwen's visual-token
        # count and can materially alter both grounding and the SAM mask.
        resized_height, resized_width = smart_resize(height, width, 28, max_pixels=400000)
        llm_image = image.resize((resized_width, resized_height))
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        sam_image = resize_api(image, (1024, 1024))
        sam_image = torch.from_numpy(np.array(sam_image, copy=True)).permute(2, 0, 1).contiguous().float()
        sam_image = (sam_image - pixel_mean) / pixel_std
        message = [
            {"role": "system", "content": self._system_template},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self._question_template.format(question=instruction)},
            ]},
        ]
        return image, llm_image, sam_image, message, (height, width), (resized_height, resized_width)

    @torch.inference_mode()
    def _run_one(self, image, instruction):
        image, llm_image, sam_image, message, orig_hw, resized_hw = self._prepare(image, instruction)
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[llm_image], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        # Qwen processors from older LENS releases emit this auxiliary key;
        # Transformers 5.x rejects it during GenerationMixin validation.
        inputs.pop("mm_token_type_ids", None)
        if self.dtype != torch.float32:
            inputs = {k: (v.to(self.dtype) if torch.is_floating_point(v) else v) for k, v in inputs.items()}
        llm_out = self.model.generate(max_new_tokens=256, use_cache=True, do_sample=False, **inputs)
        completion = self.processor.batch_decode(llm_out, skip_special_tokens=True)[0]
        attention = torch.ones_like(llm_out, dtype=torch.int64)
        if self.processor.tokenizer.pad_token_id is not None:
            attention[llm_out == self.processor.tokenizer.pad_token_id] = 0
        model_inputs = dict(inputs, input_ids=llm_out, attention_mask=attention)
        model_inputs["sam_images"] = sam_image.unsqueeze(0).to(self.device, dtype=self.dtype)
        _, low_res_mask = self.model(output_hidden_states=True, use_learnable_query=True, **model_inputs)
        mask = self.model.postprocess_masks(low_res_mask[0], orig_hw=orig_hw)
        mask = (mask[:, 0] > 0).detach().cpu().numpy()[0].astype(bool)
        box = self._parse_box(completion)
        if len(box) == 4:
            sh, sw = resized_hw
            oh, ow = orig_hw
            box = [box[0] * ow / sw, box[1] * oh / sh, box[2] * ow / sw, box[3] * oh / sh]
        else:
            box = []
        return mask, box, completion

    def generate_mask(self, images, prompt):
        images = np.asarray(images)
        masks, boxes = [], []
        for image in images:
            mask, box, _ = self._run_one(image, prompt)
            masks.append(mask)
            boxes.append(box)
        return np.stack(masks, axis=0), boxes
