# spot_ai

Natural-language command pipeline for Boston Dynamics Spot with ROS 2, schema-based execution, and camera-based perception support.

## External Dependencies

This package builds on external Spot driver stacks and does not reimplement those drivers.

- Webots Spot simulation driver: `webots_ros2_spot` from https://github.com/MASKOR/webots_ros2_spot
- Real Spot ROS 2 driver: `spot_ros2` from https://github.com/bdaiinstitute/spot_ros2

`spot_ai` provides the voice/text command pipeline, schema layer, safety gate, and adapter-specific ROS wiring on top of those repositories.

## What Lives Where

- `common core`: speech/text parsing, chat routing, command normalization, safety-policy logic
- `webots adapter`: simulation topic names, odometry, and defaults
- `real_spot adapter`: real robot topic names, odometry, and defaults

The shared core stays reusable. Only the ROS wiring changes between adapters.

## Voice Input Modes

- `mic_input_node`: use this only on a machine that has working audio input
- `wav_input_node`: use this when you already have a recorded WAV file or when WSL audio is unavailable
- `wav_input_watcher_node`: use this when Windows keeps saving recordings to the same WAV path
- `ros2 topic pub /spot_ai/voice_text`: use this for quick bypass tests

This stack is intentionally non-streaming by default. The text command is produced first, then handed to the command pipeline.

## Runtime Flow

1. Voice text enters `/spot_ai/voice_text`
2. `voice_ai_*_node` converts text into a structured command JSON and publishes it on `/spot_ai/command_json`
3. Motion-related commands pass through `schema_service_*_node` and `safety_gate_*_node`
4. Perception-related commands are routed to either `vision_caption_node` or `multi_camera_vision_node`
5. `chat_tts_node` handles spoken responses when enabled
6. The downstream Spot or Webots driver executes the validated command

## Adapter Entrypoints

- `voice_ai_webots_node`
- `voice_ai_real_spot_node`
- `schema_service_webots_node`
- `schema_service_real_spot_node`
- `safety_gate_webots_node`
- `safety_gate_real_spot_node`

Compatibility entrypoints are still available:
- `voice_ai_node`
- `schema_service_node`
- `safety_gate_node`
- `mic_input_node`
- `wav_input_node`
- `vision_caption_node`
- `multi_camera_vision_node`

## Launch Files

- `launch/spot_ai_webots.launch.py`
- `launch/spot_ai_real_spot.launch.py`

## Recommended Runs

Real Spot, single-camera perception, WAV watcher input:

```bash
ros2 launch spot_ai spot_ai_real_spot.launch.py \
  enable_mic:=false \
  enable_wav_input:=false \
  enable_wav_watcher:=true \
  enable_chat_tts:=true \
  enable_tts:=true \
  enable_multi_camera_vision:=false \
  enable_single_camera_vision:=true \
  single_camera_image_topic:=/spot/camera/hand/image/compressed \
  whisper_language:=en \
  require_wake_word:=false
```

Real Spot, multi-camera perception:

```bash
ros2 launch spot_ai spot_ai_real_spot.launch.py \
  enable_mic:=false \
  enable_wav_input:=false \
  enable_wav_watcher:=true \
  enable_chat_tts:=true \
  enable_tts:=true \
  enable_multi_camera_vision:=true \
  enable_single_camera_vision:=false \
  whisper_language:=en \
  require_wake_word:=false
```

Webots:

```bash
ros2 launch spot_ai spot_ai_webots.launch.py \
  enable_mic:=false \
  enable_chat_tts:=true \
  whisper_language:=auto \
  require_wake_word:=false
```

If you do not have audio input on the host, keep `enable_mic:=false` and feed commands with `wav_input_node`, `wav_input_watcher_node`, or `ros2 topic pub`.

## Vision Modes

- `enable_single_camera_vision:=true`: routes visual queries to `vision_caption_node`
- `single_camera_image_topic:=...`: selects the camera topic used by the single-camera pathway
- `enable_multi_camera_vision:=true`: enables `multi_camera_vision_node` for multi-view scene understanding
- `vision_query_topic`: defaults to `/spot_ai/vision_query`
- `vision_output_topic`: defaults to `/spot_ai/chat_output`

For real Spot camera-based perception:
- ensure the Spot driver publishes the required image topics
- use `/spot/camera/hand/image/compressed` for the hand-camera single-view path
- enable multi-camera mode only when the expected compressed camera topics are available

## Common Topics

- `/spot_ai/voice_text`: text command input
- `/spot_ai/command_json`: structured command output from the voice AI pipeline
- `/spot_ai/status`: short status or spoken response text
- `/spot_ai/vision_query`: perception query input
- `/spot_ai/chat_output`: chat or vision text output

## Real Spot Network Setup

Before launching the real robot stack, connect to the Spot network and export the robot address for the driver stack.

Example:

```bash
# Connect to the Boston Dynamics Wi-Fi network first.
# SSID: your_spot_wifi_ssid
# Password: your_spot_wifi_password

export SPOT_IP=your.spot.ip.address
export SPOT_USERNAME=your_spot_username
export SPOT_PASSWORD=your_spot_password
```

If your Spot driver launch uses a different set of environment variables or arguments, replace the placeholders above with the values required by your local `spot_ros2` setup.

## Build

```bash
cd /path/to/your_ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select spot_ai
source install/setup.bash
```

## Environment Variables

```bash
export GEMINI_API_KEY="<your_key>"
export GEMINI_MODEL="gemini-2.5-flash"
```

If the Gemini key is missing, the parser falls back to rule-based command parsing.

## Demo Videos

- Spot execution: https://www.youtube.com/watch?v=6kiqOmfFlxU
- Spot terminal: https://www.youtube.com/watch?v=c8kzpU7zRjs
