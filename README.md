# aero-webcam-teleop

Control a **TetherIA Aero Hand Open** robotic hand with your own hand through a webcam, in plain Python (no ROS 2).

```
webcam / RealSense D405
   │  MediaPipe hand landmarker, 21 keypoints @ ~30 Hz
   ▼
retargeting ── 16 joint angles (deg), calibrated + EMA-smoothed
   │
   ▼  control loop @ 20 Hz
aero_open_sdk ──► Aero Hand Open (ESP32-S3, USB serial)
```

It also has a **grasp mode**: press `g` and the system looks at the scene, identifies an object with YOLOv8, asks Claude which grasp to use (power, pinch, tripod, lateral pinch or palmar, at a light/medium/firm setting), and blends the hand into that pose.

Everything degrades gracefully: no hand plugged in → mock mode; no YOLO → grasp mode reports nothing found; no API key → a built-in grasp table.

## Install

Python 3.10+.

```bash
git clone https://github.com/CalebZKim/aero_hand.git
cd aero_hand
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # core: tracking + mock hand
pip install -e ".[hardware]"         # + aero-open-sdk for the real hand
pip install -e ".[grasp]"            # + YOLOv8, Anthropic SDK, python-dotenv
pip install -e ".[realsense]"        # + Intel RealSense D405 support
```

The MediaPipe hand model (`hand_landmarker.task`, ~8 MB) downloads to `~/.cache/aero_webcam_teleop/` on first run.

For LLM grasp planning, copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.

## Run

```bash
aero-webcam-teleop                          # mock mode, prints nothing unless --verbose
aero-webcam-teleop --port COM3              # Windows
aero-webcam-teleop --port /dev/ttyACM0      # Linux
aero-webcam-teleop --port COM3 --side left
aero-webcam-teleop track                    # tracking + joint bars only, no hand needed
aero-webcam-teleop home --port COM3         # tendon-safe manual homing
```

`python -m aero_webcam_teleop` works the same way.

### Keys (click the video window first)

| Key | Action |
|---|---|
| `c` | Calibrate: hold your hand open and flat, then press |
| `h` | Run the firmware homing routine at a tendon-safe speed |
| `o` | Open the hand and resume teleop (also cancels/releases a grasp) |
| `f` | Hold a fist until `o` |
| `g` | Start grasp mode; press again while holding to release |
| `a` | Toggle auto-grasp (detects every `--auto-grasp-s` seconds) |
| `q` / Esc | Quit (hand opens and the port is released) |

### Options (`run`)

| Flag | Default | |
|---|---|---|
| `--port` | none | serial port; omit for mock mode |
| `--baudrate` | 921600 | |
| `--side` | right | `right` or `left` |
| `--camera` | 0 | webcam index |
| `--fps` | 30 | capture-rate cap |
| `--control-hz` | 20 | command rate to the hand |
| `--smoothing` | 0.35 | EMA weight on the newest sample (lower = smoother, laggier) |
| `--force-webcam` | off | ignore an attached RealSense |
| `--depth-hand` | off | with a D405, build hand points from depth |
| `--auto-grasp-s` | 3.0 | auto-grasp interval |
| `--no-grasp` | off | don't load YOLO or the planner |
| `--verbose` | off | print every command in mock mode |

## Calibration

1. Start the pipeline and hold your hand **open and flat** facing the camera.
2. Press `c`.
3. Make a fist and check that the finger bars reach the top. If a finger falls short, raise its entry in `DEFAULT_JOINT_SCALE` in `retargeting.py`.

Calibration records your open-palm reading per joint and rescales so that reading maps to 0° while the full range up to each joint's limit is still reachable.

## Python API

```python
from aero_webcam_teleop import WebcamTeleopPipeline

WebcamTeleopPipeline(port="COM3", side="right").run()
```

Build your own loop from the parts:

```python
from aero_webcam_teleop import AeroHandBridge, HandTracker, Retargeter

retargeter = Retargeter(smoothing_alpha=0.4)
bridge = AeroHandBridge(port="COM3")

def on_mocap(mocap):
    bridge.send_joint_positions(retargeter.retarget(mocap))

HandTracker(hand_side="right", on_mocap=on_mocap).run()
```

## Layout

```
src/aero_webcam_teleop/
  joints.py         joint + landmark constants (single source of truth)
  tracking.py       camera backends, MediaPipe hand landmarker, HandMocap
  retargeting.py    21 landmarks -> 16 joint angles, calibration, smoothing
  bridge.py         aero_open_sdk wrapper with mock mode
  homing.py         per-actuator tendon-safe homing
  pipeline.py       state machine + control loop
  overlay.py        on-screen display
  cli.py            command line
  grasp/
    detector.py     YOLOv8 + depth-based size estimate
    planner.py      Claude grasp planner with fallback table
    poses.py        grasp pose library and force profiles
tests/              62 tests; no camera or hardware needed
```

## Joint map

16 joints in SDK order, degrees, 0 = fully extended:

| # | Joint | Range |
|---|---|---|
| 0 | thumb_cmc_abd | 0–100 |
| 1 | thumb_cmc_flex | 0–55 |
| 2 | thumb_mcp | 0–90 |
| 3 | thumb_ip | 0–90 |
| 4–6 | index mcp / pip / dip | 0–90 |
| 7–9 | middle mcp / pip / dip | 0–90 |
| 10–12 | ring mcp / pip / dip | 0–90 |
| 13–15 | pinky mcp / pip / dip | 0–90 |

The SDK maps these onto the hand's 7 actuators (thumb abduction, thumb flexion, thumb tendon, and one tendon per finger).

"Force" in grasp mode is applied as actuator **speed** (slower closing = gentler contact). The hand is position-controlled here; grip force is not measured.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Design notes

- **Tracking** uses MediaPipe's Tasks `HandLandmarker` (the legacy `mp.solutions` API, removed in MediaPipe 0.10.3x, is used only if it's all that is installed).
- **Geometry:** joint angles come from a consistent 3-D point set: MediaPipe's metric world landmarks by default, or D405 pixels deprojected with the camera intrinsics (`--depth-hand`).
- **Calibration** is computed from raw (unsmoothed) angles, so recalibrating never compounds, and it rescales rather than offsets so the full range stays reachable.
- **State machine:** manual poses hold until released, losing the hand stops commands and forces a fresh warm-up, a busy planner returns to teleop instead of stalling, and plans that arrive after a cancel are ignored.
- **Serial safety:** all SDK calls share one lock, teleop pauses during homing, and quitting opens the hand and releases the port.
- The grasp model is configurable via `AERO_GRASP_MODEL`.

## Credits

Built by **Caleb Kim**, on the TetherIA [aero-hand-open](https://github.com/TetherIA/aero-hand-open) SDK and mirrors its ROS 2 `aero_hand_open_teleop` stack.

## License

Apache 2.0, consistent with the TetherIA SDK and firmware. The Aero Hand Open hardware design files are CC BY-NC-SA 4.0.
