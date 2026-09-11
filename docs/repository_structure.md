# Repository Structure

This document describes the main runtime directories and their important entry points.

```text
model/                              Core inference pipeline
├── adarobovlg.py                   Main AdaRoboVLG model and inference entry points
├── mask_generator.py               VLM target localization and SAM3 segmentation
├── grasp_generator.py              ContactGraspNet grasp generation adapter
├── feat_projector.py               Multi-view projection and dynamic tracking
└── hand_decision/                  Multi-finger grasp scoring package
    ├── evaluator.py                Hand geometry adapter and score assignment
    ├── infer.py                    Point preprocessing and checkpoint loading
    ├── network.py                  HandDecision network and pooling head
    ├── ptv3.py                     Adapted Point Transformer V3 encoder
    └── serialization/              PTv3 space-filling curve utilities

server/                             ZMQ inference services
├── common.py                       Request validation, transport, and selection
├── visualization.py                Open3D candidate visualization
├── static_grasp/
│   ├── server_gripper.py           Static parallel-jaw service
│   └── server_dexhand.py           Static multi-finger service
├── dynamic_grasp/
│   ├── server_gripper.py           Dynamic parallel-jaw service
│   ├── server_dexhand.py           Dynamic multi-finger service
│   └── tracking.py                 Shared tracking state and pose updates
└── libero/server_gripper.py        LIBERO Franka action-chunk adapter

client/                             Reference clients for offline observation replay
├── common.py                       Packet loading and ZMQ request handling
├── static_grasp/                   Static parallel-jaw and multi-finger clients
└── dynamic_grasp/                  Dynamic parallel-jaw and multi-finger clients

dexgraspapi/                        Dexterous grasp representations and hand adaptation
├── grasp_mapping.py                Parallel-jaw to multi-finger grasp mapping
├── collision_detector.py           Multi-finger collision filtering
├── multi_finger_grasp_base.py      Shared multi-finger grasp representation
└── {allegro_hand,dh3_hand,inspire_hand}/

robots/hands/                       Robotic-hand models and geometry
├── hand_base.py                    Shared hand model and registry
├── hand_sampler_base.py            Shared surface sampling
├── hand_collision_base.py          Shared collision geometry
├── grasp_taxonomy.py               Supported grasp-type definitions
└── {allegro_hand,dh3_hand,inspire_hand}/

utils/                              Geometry, transforms, paths, and I/O helpers
test/                               Service, geometry, and inference checks
data/                               Hand assets and recorded observation examples
checkpoints/                        Local weights; Git retains documentation and manifests
docs/                               Protocol, testing, and integration documentation
third_party/                        Bundled external implementations
```

Large model and example assets under `checkpoints/` and `data/` are runtime inputs rather
than source modules. See [Request Fields](protocol.md#request-fields) for the recorded
observation schema.
