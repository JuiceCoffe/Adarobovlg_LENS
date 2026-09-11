# Model Weights

Extract the complete model-weight package to `checkpoints/` in the repository root. The
weights are hosted in the [AdaRoboVLG Hugging Face repository](https://huggingface.co/adarobovlg/adarobovlg/tree/main),
and the main README contains the expected directory layout.

| Weight | Purpose | Loaded at |
| --- | --- | --- |
| `checkpoints/sam3/` | SAM3 language-guided target segmentation | [`model/mask_generator.py`](../model/mask_generator.py) |
| `checkpoints/contact_graspnet/gc6d.pth` | ContactGraspNet parallel-jaw grasp generation | [`model/grasp_generator.py`](../model/grasp_generator.py) |
| `checkpoints/hand_decision/hand_decision_model_8145.pth` | Multi-finger grasp scoring | [`model/adarobovlg.py`](../model/adarobovlg.py) |
| `checkpoints/dinov2/dinov2_vitl14_pretrain.pth` | DINOv2 features for dynamic tracking | [`model/feat_projector.py`](../model/feat_projector.py) |
| `checkpoints/xmem/XMem.pth` | XMem mask tracking | [`model/feat_projector.py`](../model/feat_projector.py) |
| `checkpoints/xmem/resnet18-5c106cde.pth` and `resnet50-19c8e357.pth` | XMem encoder initialization | [`model/feat_projector.py`](../model/feat_projector.py) |

By default, paths are resolved relative to the repository root. Set
`ADAROBOVLG_CHECKPOINTS_DIR` to use another checkpoint directory, or override individual
paths with `ADAROBOVLG_SAM3_PATH`, `ADAROBOVLG_GRASP_CHECKPOINT`, and
`ADAROBOVLG_HAND_CHECKPOINT`.

Expected file sizes and SHA256 checksums are recorded in
[`checkpoints/manifest.json`](../checkpoints/manifest.json). Verify the files after download
before diagnosing model-loading or inference failures.
