# Third-Party Components

Use the bundled implementations because this repository depends on their local adaptations.
All five top-level components use editable registration with dependency resolution disabled;
use the commands under [Environment Installation](../README.md#environment-installation).

| Directory | Role | Source and revision | Setup | Required by documented services |
| --- | --- | --- | --- | --- |
| `contact_graspnet/` | Grasp generation and point-cloud utilities used for GraspClutter6D | [gc6d-contact-graspnet](https://github.com/SeungBack/gc6d-contact-graspnet); local revision not recorded | Editable install with `--no-deps`; bundled `ur_toolbox` is not required | All static and dynamic services |
| `d3fields/` | Multi-view projection and tracking, including XMem | [D3Fields](https://github.com/WangYixuan12/d3fields); local revision not recorded | Editable install with `--no-deps` | All services for projection; dynamic services also use tracking |
| `dinov2/` | Feature backbone for dynamic tracking | [DINOv2](https://github.com/facebookresearch/dinov2); local revision not recorded | Editable install with `--no-deps`; model construction uses the bundled source through local Torch Hub | Dynamic services only |
| `graspnetAPI/` | Grasp representations and utilities | [GraspNet API](https://github.com/graspnet/graspnetAPI); local revision not recorded | Editable install with `--no-deps` | All static and dynamic services |
| `urchin/` | URDF parsing | [Urchin](https://github.com/fishbotics/urchin), revision `dbc743e6abe6fff3b20012e487b529fe18e411b6` | Editable install with `--no-deps` | Multi-finger services |

SAM3 is loaded through Transformers from its checkpoint directory. XMem is nested within
d3fields and is not installed as a separate package. The project-owned HandDecision runtime
lives in [`model/hand_decision/`](../model/hand_decision/NOTICE.md); its PTv3-derived files
retain their upstream notice and MIT license in that directory.

## Licenses and Redistribution

Retain all accompanying license files, including those for
[GraspNet API](../third_party/graspnetAPI/LICENSE),
[d3fields](../third_party/d3fields/d3fields/LICENSE),
[XMem](../third_party/d3fields/d3fields/XMem/LICENSE),
[DINOv2](../third_party/dinov2/LICENSE), and
[urchin](../third_party/urchin/LICENSE). A package metadata license label does not establish
the licensing of every bundled component. Verify the provenance and redistribution terms of
each component, model weight, and dataset before redistribution.

The public `gc6d-contact-graspnet` repository does not currently include a standalone
license file. Do not redistribute this vendored directory until the project maintainer has
confirmed permission or supplied the applicable license terms.
