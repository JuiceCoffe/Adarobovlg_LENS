# Troubleshooting

## Shutting Down a Service

Press Ctrl+C in the server terminal so the process can release its window, socket, and GPU resources. Use `ss -ltnp` to check whether a port remains occupied. Avoid terminating processes by obsolete script names because that can affect unrelated jobs.

## Common Problems

| Symptom | Checks |
| --- | --- |
| A third-party package cannot be imported | Run from the repository root, activate the correct environment, and install the bundled packages as described in the main README. |
| CUDA or an extension fails to load | Verify that PyTorch, CUDA, DGL, PyTorch3D, scatter, and cluster use compatible builds. Do not upgrade PyTorch alone. |
| A checkpoint is missing | Download the weights from the [AdaRoboVLG Hugging Face repository](https://huggingface.co/adarobovlg/adarobovlg/tree/main), then check their paths, configuration files, and tokenizer against the model-weight documentation. |
| The API key is missing | Set `DASHSCOPE_API_KEY` for the default remote box-localization mode, or use `--text-prompt` for local English text segmentation. |
| Workspace filtering removes every point | Verify the world coordinate frame, meter/millimeter conversion, and `--workspace-bounds`. |
| Points remain but no grasp is returned | Inspect target segmentation, hand mapping, and final hand-base bounds before expanding the workspace. |
| The client waits indefinitely | Check whether the server is computing or waiting for `select <index>`. Dynamic initialization requires terminal confirmation even with `--no-visualizer`. Closing the window cancels an active selection but only disables display after tracking starts. |
| Camera-pose or image validation fails | Ensure all camera arrays share the same view order, images are uint8 BGR, and depths are finite, nonnegative, and expressed in meters. |
| The tracking ID does not match | Use the ID returned by the current server process. Restarting the service or initializing a new target invalidates the previous ID. |
| Tracking leaves the workspace or returns `requires_reinit=True` | Initialize again with a new observation; later frames cannot restore an invalidated session. |
| A request times out | Close and recreate the REQ socket without automatically resending. A client timeout does not cancel server-side processing. |
| The Open3D window does not start | Verify the desktop display environment or use `--no-visualizer`. Headless checks do not validate window interaction. |
| Numba or Matplotlib cannot write its cache | Set `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to directories writable by the current user. |

See the [protocol](protocol.md) for status and error fields. When reporting a problem, include the command, package versions, stack trace, and array shapes, but never include credentials or private calibration data.
