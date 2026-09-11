# Testing

The scripts under `test/` are regression checks for interfaces, geometry, service behavior, and processing logic. They are not benchmarks or evaluations of the grasping performance reported in the paper.

Run all commands from the repository root with the `AdaRoboVLG` environment activated.

## Regression Checks

The default checks use deterministic fixtures or mocked model calls. They require the project dependencies, but do not require model weights, a GPU, recorded observations, a VLM service, or a display window.

```bash
python test/test_checkpoint_paths.py
python test/test_code_cleanup.py
python test/test_gripper_alignment.py
python test/test_lazy_hand_models.py
python test/test_libero_server.py
python test/test_release_hygiene.py
python test/test_server_imports.py
python test/test_service_interaction.py
python test/test_target_services.py
python test/test_target_tracking.py
python test/test_workspace_bounds.py
```

| Script | Coverage |
| --- | --- |
| `test_checkpoint_paths.py` | Checkpoint lookup, overrides, and missing-file errors |
| `test_code_cleanup.py` | Hand outputs, segmentation fallback, and JSON encoding |
| `test_gripper_alignment.py` | Depth offsets and dynamic orientation constraints |
| `test_lazy_hand_models.py` | Lazy loading and per-hand model caching |
| `test_libero_server.py` | LIBERO validation, action conversion, errors, and ZMQ compatibility |
| `test_release_hygiene.py` | Personal paths, generated files, example manifest, and README anchors |
| `test_server_imports.py` | Import isolation for all service entry points |
| `test_service_interaction.py` | Selection, error handling, sessions, and client requests |
| `test_target_services.py` | Static gripper and multi-finger service responses |
| `test_target_tracking.py` | Dynamic session state and rigid pose updates |
| `test_workspace_bounds.py` | Input/output workspace filtering and field alignment |

A successful script prints one or more `PASS` messages and exits with status 0.

## Real Inference Check

`test_target_services.py` also provides an optional static inference check. It requires a compatible GPU, the model weights, and a trusted request packet matching the documented protocol and workspace bounds.

Before GPU inference, verify that the packaged HandDecision architecture matches the
released checkpoint exactly:

```bash
python test/test_hand_decision_package.py
```

Use local English text segmentation:

```bash
python test/test_target_services.py \
  --data data/examples/static/red_chips_can.pkl \
  --prompt 'red chips can' --task both
```

Use VLM box localization instead:

```bash
export DASHSCOPE_API_KEY='YOUR_API_KEY'
python test/test_target_services.py \
  --data data/examples/static/red_chips_can.pkl \
  --prompt 'red chips can' --task both --box-prompt
```

Set `--task gripper` or `--task hand` to check one service type. For the multi-finger check, select `AllegroHand`, `InspireHand`, or `DH3Hand` with `--hand`.

## ZMQ Service Check

Add `--network` to start each selected static service in a child process and validate a real local ZMQ request/reply:

```bash
python test/test_target_services.py \
  --data data/examples/static/red_chips_can.pkl \
  --prompt 'red chips can' --task both --network
```

The test uses a temporary local port and stops the service after receiving the reply. Add `--box-prompt` if VLM localization is required.

Only load trusted pickle files and connect to trusted endpoints. See [Protocol](protocol.md) for packet requirements and [Troubleshooting](troubleshooting.md) for common runtime errors.
