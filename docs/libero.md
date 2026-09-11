# AdaRoboVLG-Playground Evaluation

[AdaRoboVLG-Playground](https://github.com/AdaRoboVLG/AdaRoboVLG-Playground) is built on
LIBERO to test language-guided target grasping with a parallel-jaw gripper in static scenes.
Its simulator client connects to the AdaRoboVLG adapter over ZMQ. The model server receives
two-view RGB-D observations, a fused world-frame point cloud, and a language target, then
returns a 74-step absolute Franka end-effector action chunk.

This is the target-grasp evaluation protocol used for the paper. The companion evaluator
adapts the selected LIBERO tasks by replacing their language instructions with target-object
grasp prompts and their success conditions with `Grasped` predicates. Tasks marked
`invalid` are skipped rather than evaluated as long-horizon manipulation tasks.

AdaRoboVLG-Playground is adapted from
[GraspVLA-playground](https://github.com/MiYanDoris/GraspVLA-playground) and contains two
independent evaluations:

- **LIBERO target grasping:** the grasp-only Long, Goal, and Object suites used for model
  comparison.
- **Objaverse Playground:** randomized scenes containing six sampled Objaverse objects,
  layouts, targets, and background textures.

## Simulator Installation

The model server and simulator require separate environments. Install the model environment
from the main [Environment Installation](../README.md#environment-installation) section. Use
the following steps for the simulator/client environment.

### 1. System Dependencies and Repository

```bash
sudo apt install git-lfs ffmpeg

cd <your-ws-path>
git clone https://github.com/AdaRoboVLG/AdaRoboVLG-Playground.git
cd AdaRoboVLG-Playground
```

### 2. Python Environment and PyTorch

```bash
conda create -n playground python=3.9.0
conda activate playground

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"

pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 3. Vendored robosuite and BDDL

```bash
cd <your-ws-path>/AdaRoboVLG-Playground/third_party/robosuite
pip install -e .

cd <your-ws-path>/AdaRoboVLG-Playground/third_party/bddl
pip install -e .
```

The bundled `libero/` package is loaded directly when commands are run from the Playground
repository root; do not install another LIBERO copy.

### 4. cuRobo

The simulator client uses cuRobo for Franka forward kinematics and is tested with cuRobo
v0.7.4.

```bash
cd <your-ws-path>
git clone --branch v0.7.4 --depth 1 https://github.com/NVlabs/curobo.git curobo-v0.7.4
cd curobo-v0.7.4
git lfs pull
pip install -e . --no-build-isolation
```

### 5. Asset and LIBERO Paths

Replace `<your-ws-path>` in `assets/franka_with_extended_finger/franka.yml` with the absolute
workspace path, then initialize LIBERO's local path configuration:

```bash
cd <your-ws-path>/AdaRoboVLG-Playground
python -c "from libero.libero import set_libero_default_path; set_libero_default_path()"
```

For a headless machine, set:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
```

## Evaluation Workflow

Run all commands from the corresponding repository root.

### 1. Start and Check the AdaRoboVLG Server

```bash
# Terminal 1: AdaRoboVLG environment
conda activate AdaRoboVLG
cd <your-ws-path>/AdaRoboVLG
python server/libero/server_gripper.py
```

The server listens on `tcp://127.0.0.1:6666` by default. It uses VLM localization unless
`--text-prompt` is supplied; configure the VLM variables described in the main README.

The following command checks transport availability; it does not run complete grasp
inference:

```bash
# Terminal 2: simulator environment
conda activate playground
cd <your-ws-path>/AdaRoboVLG-Playground
python validate_server.py --host 127.0.0.1 --port 6666
```

An error reply is also a dictionary, so confirm successful initialization in the model-server
terminal.

### 2. Evaluate LIBERO

Run 50 trials for each configured target-grasp task:

```bash
python evaluate_libero_tasks.py name=libero_tasks trial_num=50
```

Or split the same trials across five simulator processes. The generated clients use the
server port configured in `config/parallel_eval.yaml`.

```bash
python parallel_eval.py \
  exp_name=parallel_eval_libero_tasks \
  parallel_env_num=5 \
  trial_num=50
```

The default configuration enables `libero_object`, `libero_10`, and `libero_goal`, with at
most ten tasks per suite. `Long` corresponds to `libero_10`. Change suites, trial count,
task limit, and ports through the Hydra files under `config/` or command-line overrides.

#### LIBERO Success Rate (%)

| Method | Long | Goal | Object |
| --- | ---: | ---: | ---: |
| [OpenVLA](https://arxiv.org/abs/2406.09246) (fine-tuned) | 33.7 | 56.6 | 65.4 |
| [π₀](https://arxiv.org/abs/2410.24164) (fine-tuned) | 62.7 | 79.4 | 93.8 |
| [GraspVLA](https://arxiv.org/abs/2505.03233) (zero-shot) | 82.0 | 91.2 | 94.1 |
| [AnyGrasp](https://arxiv.org/abs/2212.08333)* (zero-shot) | 87.7 | 90.3 | 91.6 |
| AdaRoboVLG (zero-shot) | **93.1** | **92.6** | **97.4** |

The OpenVLA, π₀, and GraspVLA values are reported in the
[GraspVLA paper](https://arxiv.org/pdf/2505.03233). AnyGrasp* denotes an enhanced AnyGrasp
variant using the same VLM and SAM3 target-localization and segmentation components as
AdaRoboVLG.

These values measure target grasping after eligible LIBERO instructions are simplified to
`pick up {object}`; they are not success rates for the original multi-stage LIBERO tasks.
Unlike the latest GraspVLA Playground client, this evaluator does not lift, open the gripper,
and clear proprioceptive history after a failed grasp.

Aggregate a single-process run with:

```bash
python misc/get_statistics.py name=libero_tasks
```

### 3. Evaluate the Objaverse Playground

Start with a one-object smoke test:

```bash
python playground.py name=playground_smoke trial_num=1 object_num=1
```

Run the six-object setting:

```bash
python playground.py name=playground_test trial_num=10 object_num=6
```

For reproducible scenes, provide a saved seed list:

```bash
python playground.py name=playground_fixed \
  object_num=6 \
  seed_list_file=config/example_seeds.json
```

Calculate overall and per-object success rates:

```bash
python misc/get_success_rate_per_object.py \
  --result_path data/playground_test
```

Playground results are reported separately from the LIBERO results.

## Outputs and Execution Notes

Evaluation artifacts are written below `data/<name>/`; videos ending in `_success.mp4` are
counted as successful episodes. Statistics for a single-process LIBERO run are written to
`data/<name>/statistics.txt`. Skipped `invalid` tasks do not create videos or enter the
denominator. The parallel launcher performs equivalent aggregation automatically.

All simulator processes use the configured model-server port. One ZMQ REP server accepts
multiple clients but handles model requests serially; parallel evaluation accelerates the
simulator side, not model inference. The parallel launcher must reference the actual
simulator conda environment name.

## Request Format

The trusted request is a pickled Python dictionary. The adapter consumes these fields:

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `prompt` | nonempty `str` | Target description after the client removes the `pick up ` prefix. |
| `points` | float `(N,3)` | Fused world-frame point cloud in meters. |
| `colors` | float `(N,3)` | RGB point colors in `[0,1]`. |
| `obs.color` | `uint8 (2,H,W,3)` | Front and side images in the evaluator's channel convention. |
| `obs.depth` | float `(2,H,W)` | Metric depth aligned with the images. |
| `obs.K` | float `(2,3,3)` | Camera intrinsics. |
| `obs.pose` | float `(2,3,4)` or `(2,4,4)` | World-to-camera extrinsics used for projection. |
| `obs.world2robot` | float `(4,4)` | Despite its legacy name, the client supplies `T_world_from_robot`. |

Legacy packet fields such as `front_view_image`, `side_view_image`, `proprio_array`, `text`,
and `images` are not used by this adapter. Pickle can execute Python reconstruction code;
connect only trusted clients to the service.

## Reply and Action Chunk

A successful reply has this form:

```python
{
    "result": [action_0, ..., action_73],
    "mode": "abs",
    "debug": {"bbox": np.ndarray((2, 4), dtype=np.float32)},
}
```

Each action is `[x, y, z, roll, pitch, yaw, gripper]` in the Franka base frame. Position is
in meters, angles are in radians, `1` opens the model-side gripper, and `-1` closes it. The
client reverses the gripper sign before passing an action to robosuite.

The fixed action chunk contains 20 high pre-grasp actions, 15 local pre-grasp actions, 15
open-gripper grasp actions, 4 closed-gripper grasp actions, and 20 closed-gripper lift
actions. This behavior matches the adapted target-grasp success conditions used by the
evaluation harness.

Errors contain `error` and `error_code="request_failed"` without a `result`. The existing
client holds the current robot pose for one step and requests another prediction.
