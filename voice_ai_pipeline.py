import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .spot_schema_names import PRIMITIVE_TO_SNIPPET, SCHEMA_COMMANDS
from .motor_schema import MOTOR_SCHEMA_EXECUTORS
from spot_schemas_interfaces.srv import Walk, Rotate, GraspHand, ReleaseHand

try:
    import google.generativeai as genai
except Exception:
    genai = None


SYSTEM_PROMPT = """
You are Spot command parser. Convert user command into STRICT JSON only.
Allowed primitives:
- WalkForward: Walk straight forward by x distance
- WalkBackward: Walk straight backward by x distance
- WalkLeft: Side-step left by x distance
- WalkRight: Side-step right by x distance
- Stop: Immediate stop ongoing task/motion
- RotateLeft: Rotate in place counter-clockwise by angle_degrees
- RotateRight: Rotate in place clockwise by angle_degrees
- Stand: Move Spot to standing posture
- Sit: Move Spot to sitting posture
- ArmUnstow: Move arm to ready position
- ArmStow: Fold arm into stowed position
- GraspHand: Close gripper
- ReleaseHand: Open gripper
- ExtendArm: Extend arm by x,y,z target pose

Response style: assistant_response must be short (max 1 sentence, <= 40 chars).

JSON schema:
{
  "header": {
    "timestamp": "ISO-8601",
    "robot_id": "spot-01",
    "pipeline_version": "v1"
  },
  "perception": {
    "input_source": "speech",
    "raw_input": "string",
    "interpretation": {
      "category": "locomotion|manipulation|posture",
      "name": "one of allowed primitives",
      "confidence": 0.0
    }
  },
  "behavior_execution": {
    "primitive": "one of allowed primitives",
    "priority": "normal",
    "parameters": {
      "distance_m": 0.0,
      "speed_mps": 0.0,
      "angle_degrees": 0.0,
      "angular_speed_rps": 0.0,
      "x": 0.0,
      "y": 0.0,
      "z": 0.0
    }
  },
  "snippet": {
    "name": "snake_case primitive name",
    "args": {}
  },
  "assistant_response": "short sentence"
}

Rules:
- For locomotion command, fill distance/speed or angle as needed.
- For manipulation command, fill relevant parameters.
- If user intent is unclear, choose Stop.
- Return JSON only. No markdown.
""".strip()


class SpotVoiceAIPipeline(Node):
    def __init__(self, parameter_defaults: Optional[dict] = None) -> None:
        super().__init__("spot_voice_ai_pipeline")

        defaults = parameter_defaults or {}

        self.declare_parameter("voice_input_topic", defaults.get("voice_input_topic", "/spot_ai/voice_text"))
        self.declare_parameter("command_json_topic", defaults.get("command_json_topic", "/spot_ai/command_json"))
        self.declare_parameter("status_topic", defaults.get("status_topic", "/spot_ai/status"))
        self.declare_parameter("vision_query_topic", defaults.get("vision_query_topic", "/spot_ai/vision_query"))
        self.declare_parameter("chat_input_topic", defaults.get("chat_input_topic", "/spot_ai/chat_input"))
        self.declare_parameter("cmd_vel_topic", defaults.get("cmd_vel_topic", "/spot_ai/cmd_vel_raw"))
        self.declare_parameter("require_wake_word", defaults.get("require_wake_word", True))
        self.declare_parameter("wake_words", defaults.get("wake_words", ["spot", "hey spot", "ok spot", "hello spot"]))
        self.declare_parameter("wake_followup_window_sec", defaults.get("wake_followup_window_sec", 6.0))
        self.declare_parameter("wake_ack_text", defaults.get("wake_ack_text", "Yes?"))
        self.declare_parameter("enable_tts", defaults.get("enable_tts", True))
        self.declare_parameter("tts_language", defaults.get("tts_language", "en"))
        self.declare_parameter("tts_player_cmd", defaults.get("tts_player_cmd", "mpg123"))
        self.declare_parameter("walk_duration_scale", defaults.get("walk_duration_scale", 1.0))
        self.declare_parameter("sim_walk_duration_scale", defaults.get("sim_walk_duration_scale", 8.0))
        self.declare_parameter("auto_sim_scale", defaults.get("auto_sim_scale", True))
        self.declare_parameter("rotation_duration_scale", defaults.get("rotation_duration_scale", 1.0))
        self.declare_parameter("allow_windows_popup_fallback", defaults.get("allow_windows_popup_fallback", False))
        self.declare_parameter("enable_heading_hold", defaults.get("enable_heading_hold", True))
        self.declare_parameter("heading_hold_use_sim_only", defaults.get("heading_hold_use_sim_only", True))
        self.declare_parameter("heading_hold_kp", defaults.get("heading_hold_kp", 1.2))
        self.declare_parameter("heading_hold_max_angular_z", defaults.get("heading_hold_max_angular_z", 0.35))
        self.declare_parameter("odom_topic", defaults.get("odom_topic", "/Spot/odometry"))
        self.declare_parameter(
            "tts_tmp_file",
            defaults.get("tts_tmp_file", "/mnt/c/Users/jaeyk/Desktop/Spot/LLM-SPOT-Project/speech.mp3"),
        )
        self.declare_parameter("auto_play_windows", defaults.get("auto_play_windows", True))
        self.declare_parameter("ignore_recent_tts_echo_sec", defaults.get("ignore_recent_tts_echo_sec", 4.0))
        self.declare_parameter("enable_hand_camera_macro", defaults.get("enable_hand_camera_macro", True))
        self.declare_parameter(
            "hand_camera_vision_query",
            defaults.get("hand_camera_vision_query", "What do you see in front of you?"),
        )
        self.declare_parameter(
            "hand_camera_vision_delay_sec",
            defaults.get("hand_camera_vision_delay_sec", 1.0),
        )

        voice_input_topic = str(self.get_parameter("voice_input_topic").value)
        command_json_topic = str(self.get_parameter("command_json_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        vision_query_topic = str(self.get_parameter("vision_query_topic").value)
        chat_input_topic = str(self.get_parameter("chat_input_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)

        self.voice_sub = self.create_subscription(
            String, voice_input_topic, self._voice_text_cb, 10
        )
        self.json_pub = self.create_publisher(String, command_json_topic, 10)
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.vision_query_pub = self.create_publisher(String, vision_query_topic, 10)
        self.chat_input_pub = self.create_publisher(String, chat_input_topic, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.stand_client = self.create_client(Trigger, "/stand")
        self.sit_client = self.create_client(Trigger, "/sit")
        self.arm_unstow_client = self.create_client(Trigger, "/arm_unstow")
        self.arm_stow_client = self.create_client(Trigger, "/arm_stow")
        self.close_gripper_client = self.create_client(Trigger, "/close_gripper")
        self.open_gripper_client = self.create_client(Trigger, "/open_gripper")

        # Schema service clients
        self.walk_client = self.create_client(Walk, '/schemas/walk')
        self.rotate_client = self.create_client(Rotate, '/schemas/rotate')
        self.grasp_client = self.create_client(GraspHand, '/schemas/grasp_hand')
        self.release_client = self.create_client(ReleaseHand, '/schemas/release_hand')

        self.motion_timer = self.create_timer(0.1, self._motion_tick)
        self.motion_active = False
        self.motion_end_sec = 0.0
        self.motion_linear_x = 0.0
        self.motion_linear_y = 0.0
        self.motion_angular_z = 0.0
        self.pending_motion_commands = []
        self.current_yaw: Optional[float] = None
        self.target_yaw: Optional[float] = None
        self.heading_hold_enabled_for_motion = False
        self._tts_warned = False
        self.last_spoken_text = ""
        self.last_spoken_time_sec = 0.0
        self.awaiting_followup_until_sec = 0.0
        # Hold one-shot timers so they don't get garbage collected.
        self._macro_timers = []

        self.enable_tts = bool(self.get_parameter("enable_tts").value)
        self.tts_language = str(self.get_parameter("tts_language").value)
        self.tts_player_cmd = str(self.get_parameter("tts_player_cmd").value)
        self.rotation_duration_scale = float(self.get_parameter("rotation_duration_scale").value)
        base_scale = float(self.get_parameter("walk_duration_scale").value)
        sim_scale = float(self.get_parameter("sim_walk_duration_scale").value)
        auto_sim_scale = bool(self.get_parameter("auto_sim_scale").value)
        use_sim_time = (
            bool(self.get_parameter("use_sim_time").value)
            if self.has_parameter("use_sim_time")
            else False
        )
        self.is_sim_time = use_sim_time
        self.walk_duration_scale = sim_scale if (auto_sim_scale and use_sim_time) else base_scale
        self.allow_windows_popup_fallback = bool(
            self.get_parameter("allow_windows_popup_fallback").value
        )
        self.enable_heading_hold = bool(self.get_parameter("enable_heading_hold").value)
        self.heading_hold_use_sim_only = bool(
            self.get_parameter("heading_hold_use_sim_only").value
        )
        self.heading_hold_kp = float(self.get_parameter("heading_hold_kp").value)
        self.heading_hold_max_angular_z = float(
            self.get_parameter("heading_hold_max_angular_z").value
        )
        self.tts_tmp_file = str(self.get_parameter("tts_tmp_file").value)
        self.auto_play_windows = bool(self.get_parameter("auto_play_windows").value)
        self.ignore_recent_tts_echo_sec = float(
            self.get_parameter("ignore_recent_tts_echo_sec").value
        )
        self.enable_hand_camera_macro = bool(self.get_parameter("enable_hand_camera_macro").value)
        self.hand_camera_vision_query = str(
            self.get_parameter("hand_camera_vision_query").value
        ).strip()
        self.hand_camera_vision_delay_sec = float(
            self.get_parameter("hand_camera_vision_delay_sec").value
        )
        self.require_wake_word = bool(self.get_parameter("require_wake_word").value)
        wake_words = self.get_parameter("wake_words").value
        self.wake_words = [
            self._normalize_text(word)
            for word in (wake_words if isinstance(wake_words, list) else [])
            if self._normalize_text(word)
        ]
        self.wake_followup_window_sec = float(
            self.get_parameter("wake_followup_window_sec").value
        )
        self.wake_ack_text = str(self.get_parameter("wake_ack_text").value).strip() or "Yes?"

        self.model = self._init_model()
        self.get_logger().info(
            f"spot_voice_ai_pipeline ready: in={voice_input_topic} cmd={cmd_vel_topic} sim_scale={self.walk_duration_scale:.2f}"
        )
        self.cmd_vel_topic = cmd_vel_topic

    def _call_trigger_service(self, client, service_name: str, action_name: str) -> bool:
        if client is None:
            self.get_logger().warning(f"{action_name} unavailable: {service_name} client not created")
            return False
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"{action_name} unavailable: service {service_name} not ready")
            return False

        future = client.call_async(Trigger.Request())

        def _done_callback(done_future):
            try:
                response = done_future.result()
            except Exception as exc:
                self.get_logger().error(f"{action_name} call failed: {exc}")
                return

            if response.success:
                message = response.message or "ok"
                self.get_logger().info(f"{action_name} accepted: {message}")
            else:
                message = response.message or "unknown error"
                self.get_logger().warning(f"{action_name} rejected: {message}")

        future.add_done_callback(_done_callback)
        return True

    def _init_model(self):
        api_key = os.getenv("GEMINI_API_KEY")
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not genai:
            self.get_logger().warning("google.generativeai unavailable, using fallback parser")
            return None
        if not api_key:
            self.get_logger().warning("GEMINI_API_KEY not set, using fallback parser")
            return None

        try:
            genai.configure(api_key=api_key)
            for model_name in [preferred_model, "gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.get_logger().info(f"Gemini model enabled: {model_name}")
                    return model
                except Exception:
                    continue
            raise RuntimeError("No usable Gemini model from preferred list")
        except Exception as exc:
            self.get_logger().warning(f"Gemini init failed, fallback parser enabled: {exc}")
            return None

    def _voice_text_cb(self, msg: String) -> None:
        raw_text = msg.data.strip()
        if not raw_text:
            return

        self.get_logger().info(f"Heard: {raw_text}")

        if self._is_recent_tts_echo(raw_text):
            self.get_logger().info(f"Ignoring likely TTS echo: {raw_text}")
            return

        activated_text = self._extract_activated_text(raw_text)
        if activated_text is None:
            self.get_logger().info(f"Ignoring input without wake word: {raw_text}")
            return
        if not activated_text:
            self._acknowledge_wake_word()
            return
        raw_text = activated_text

        self.get_logger().info(f"Input: {raw_text}")

        if self._handle_hand_camera_macro(raw_text):
            return

        # Route natural-language vision requests to the vision node.
        if self._should_route_to_vision(raw_text):
            vision_msg = String()
            vision_msg.data = raw_text
            self.vision_query_pub.publish(vision_msg)
            self.get_logger().info("Routed input to vision query topic")
            return

        if self._should_route_to_chat(raw_text):
            chat_msg = String()
            chat_msg.data = raw_text
            self.chat_input_pub.publish(chat_msg)
            self.get_logger().info("Routed input to chat topic")
            return

        commands = self._build_command_sequence(raw_text)
        if len(commands) > 1:
            self.get_logger().info(f"Parsed compound command into {len(commands)} steps")
        self._execute_commands(commands)
        for command in commands:
            self._publish_outputs(command)

    def _handle_hand_camera_macro(self, raw_text: str) -> bool:
        if not self.enable_hand_camera_macro:
            return False

        normalized = self._normalize_text(raw_text)
        if "hand camera" not in normalized:
            return False
        if not any(keyword in normalized for keyword in ["look", "see", "show"]):
            return False

        # Open the hand first, then query vision shortly after.
        command = self._build_command_json("open hand")
        self._execute_commands([command])
        self._publish_outputs(command)

        query = self.hand_camera_vision_query or "What do you see in front of you?"
        self._schedule_vision_query(self.hand_camera_vision_delay_sec, query)
        self.get_logger().info("Triggered hand camera macro: open hand -> vision query")
        return True

    def _schedule_vision_query(self, delay_sec: float, query: str) -> None:
        delay = max(0.0, float(delay_sec))
        if delay <= 0.0:
            msg = String()
            msg.data = query
            self.vision_query_pub.publish(msg)
            self.get_logger().info("Hand camera macro: vision query published")
            return

        timer_ref = {"timer": None}

        def _fire() -> None:
            msg = String()
            msg.data = query
            self.vision_query_pub.publish(msg)
            self.get_logger().info("Hand camera macro: vision query published")
            if timer_ref["timer"] is not None:
                timer_ref["timer"].cancel()
                try:
                    self._macro_timers.remove(timer_ref["timer"])
                except ValueError:
                    pass

        timer_ref["timer"] = self.create_timer(delay, _fire)
        self._macro_timers.append(timer_ref["timer"])

    def _build_command_sequence(self, raw_text: str) -> list[dict]:
        segments = self._split_compound_command(raw_text)
        commands = []
        for segment in segments:
            cleaned = segment.strip()
            if not cleaned:
                continue
            commands.append(self._build_command_json(cleaned))
        return commands or [self._build_command_json(raw_text)]

    def _build_command_json(self, raw_text: str) -> dict:
        fallback = self._fallback_parse(raw_text)
        if self._should_use_fast_path(raw_text):
            self.get_logger().info("Using the simple parser for a short locomotion command")
            return fallback

        if not self.model:
            return fallback

        try:
            prompt = f"{SYSTEM_PROMPT}\n\nraw_input: {raw_text}\nReturn strict JSON now."
            response = self.model.generate_content(prompt)
            candidate = self._extract_json(response.text if hasattr(response, "text") else "")
            parsed = json.loads(candidate)
            return self._normalize_command(parsed, raw_text)
        except Exception as exc:
            self.get_logger().warning(f"AI parse failed, fallback parser used: {exc}")
            return fallback

    def _fallback_parse(self, raw_text: str) -> dict:
        lowered = raw_text.lower().strip()
        primitive = self._classify_primitive(lowered)
        distance = self._extract_distance_m(lowered, default=1.0)
        distance = max(0.1, min(distance, 5.0))
        duration_sec = self._extract_duration_seconds(lowered, default=0.0)
        angle_degrees = self._extract_angle_degrees(lowered, default=90.0)
        angle_degrees = max(5.0, min(angle_degrees, 360.0))
        speed = 0.45
        angular_speed_rps = 0.6

        if any(word in lowered for word in ["fast", "quick"]):
            speed = 0.7
            angular_speed_rps = 0.9
        elif any(word in lowered for word in ["slow"]):
            speed = 0.25
            angular_speed_rps = 0.35

        if (
            duration_sec > 0.0
            and primitive in {"WalkForward", "WalkBackward", "WalkLeft", "WalkRight"}
            and not re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|meters)\b", lowered)
        ):
            distance = max(0.1, min(duration_sec * speed, 5.0))

        if primitive == "Stop":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Stopping now."
        elif primitive == "WalkForward":
            answer = "Moving forward."
        elif primitive == "WalkBackward":
            answer = "Moving backward."
        elif primitive == "WalkLeft":
            answer = "Moving left."
        elif primitive == "WalkRight":
            answer = "Moving right."
        elif primitive == "RotateLeft":
            distance = 0.0
            speed = 0.0
            answer = "Turning left."
        elif primitive == "RotateRight":
            distance = 0.0
            speed = 0.0
            answer = "Turning right."
        elif primitive == "Stand":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Standing by."
        elif primitive == "Sit":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Sitting down."
        elif primitive == "ArmUnstow":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Unstowing arm."
        elif primitive == "ArmStow":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Stowing arm."
        elif primitive == "GraspHand":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Closing gripper."
        elif primitive == "ReleaseHand":
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Opening gripper."
        else:
            primitive = "Stop"
            distance = 0.0
            speed = 0.0
            angle_degrees = 0.0
            angular_speed_rps = 0.0
            answer = "Stopping now."

        return self._command_dict(
            raw_text=raw_text,
            primitive=primitive,
            confidence=0.72,
            distance=distance,
            speed=speed,
            angle_degrees=angle_degrees,
            angular_speed_rps=angular_speed_rps,
            assistant_response=answer,
        )

    def _normalize_command(self, data: dict, raw_text: str) -> dict:
        primitive_raw = (
            data.get("behavior_execution", {}).get(
                "primitive",
                data.get("perception", {}).get("interpretation", {}).get("name", "Stop"),
            )
        )
        primitive = self._canonical_primitive(str(primitive_raw))
        if primitive not in SCHEMA_COMMANDS:
            primitive = "Stop"

        params = data.get("behavior_execution", {}).get("parameters", {})
        text_distance = self._extract_distance_m(raw_text, default=1.0)
        text_angle = self._extract_angle_degrees(raw_text, default=90.0)

        distance = self._safe_float(params.get("distance_m", text_distance), text_distance)
        speed = self._safe_float(params.get("speed_mps", 0.45), 0.45)
        angle = self._safe_float(params.get("angle_degrees", text_angle), text_angle)
        angular_speed = self._safe_float(params.get("angular_speed_rps", 0.6), 0.6)
        x = self._safe_float(params.get("x", 0.0), 0.0)
        y = self._safe_float(params.get("y", 0.0), 0.0)
        z = self._safe_float(params.get("z", 0.0), 0.0)

        if primitive == "Stop":
            distance = 0.0
            speed = 0.0
            angle = 0.0
            angular_speed = 0.0
        elif primitive in {"WalkForward", "WalkBackward", "WalkLeft", "WalkRight"}:
            distance = max(0.1, min(distance, 5.0))
            speed = max(0.15, min(speed, 1.0))
            angle = 0.0
        elif primitive in {"RotateLeft", "RotateRight"}:
            distance = 0.0
            speed = 0.0
            angle = max(5.0, min(angle, 360.0))
            angular_speed = max(0.2, min(angular_speed, 1.5))
        else:
            distance = 0.0
            speed = 0.0
            angle = 0.0
            angular_speed = 0.0

        short_text = str(data.get("assistant_response", "Executing command.")).strip()[:40]
        if not short_text:
            short_text = self._default_response_for_primitive(primitive)

        confidence = self._safe_float(
            data.get("perception", {}).get("interpretation", {}).get("confidence", 0.8),
            0.8,
        )
        return self._command_dict(
            raw_text=raw_text,
            primitive=primitive,
            confidence=confidence,
            distance=distance,
            speed=speed,
            angle_degrees=angle,
            angular_speed_rps=angular_speed,
            x=x,
            y=y,
            z=z,
            assistant_response=short_text,
        )

    def _command_dict(
        self,
        *,
        raw_text: str,
        primitive: str,
        confidence: float,
        distance: float,
        speed: float,
        angle_degrees: float,
        angular_speed_rps: float,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        assistant_response: str,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        snippet_name = PRIMITIVE_TO_SNIPPET.get(primitive, "stop")
        category = self._category_for_primitive(primitive)
        return {
            "header": {
                "timestamp": now,
                "robot_id": "spot-01",
                "pipeline_version": "v1",
            },
            "perception": {
                "input_source": "speech",
                "raw_input": raw_text,
                "interpretation": {
                    "category": category,
                    "name": primitive,
                    "confidence": float(confidence),
                },
            },
            "behavior_execution": {
                "primitive": primitive,
                "priority": "normal",
                "parameters": {
                    "distance_m": float(distance),
                    "speed_mps": float(speed),
                    "angle_degrees": float(angle_degrees),
                    "angular_speed_rps": float(angular_speed_rps),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                },
            },
            "snippet": {
                "name": snippet_name,
                "args": {
                    "distance_m": float(distance),
                    "speed_mps": float(speed),
                    "angle_degrees": float(angle_degrees),
                    "angular_speed_rps": float(angular_speed_rps),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                },
            },
            "assistant_response": assistant_response.strip()[:40] or "Executing command.",
        }

    def _execute_command(self, command: dict, *, force_local: bool = False) -> None:
        primitive = command.get("behavior_execution", {}).get("primitive", "")
        params = command.get("behavior_execution", {}).get("parameters", {})

        self.get_logger().info(
            "Executing command: "
            f"primitive={primitive}, distance_m={float(params.get('distance_m', 0.0)):.2f}, "
            f"speed_mps={float(params.get('speed_mps', 0.0)):.2f}, "
            f"angle_deg={float(params.get('angle_degrees', 0.0)):.2f}"
        )

        # Try service first
        if not force_local and self._try_execute_via_service(primitive, params):
            return

        # Fallback to snippet
        self.get_logger().warning(f"Service failed for {primitive}, falling back to snippet")
        self._execute_snippet(command)

    def _execute_commands(self, commands: list[dict]) -> None:
        if not commands:
            return

        if len(commands) == 1:
            self.pending_motion_commands = []
            self._execute_command(commands[0])
            return

        first, rest = commands[0], commands[1:]
        self.pending_motion_commands = list(rest)
        self._execute_command(first, force_local=True)

    def _try_execute_via_service(self, primitive: str, params: dict) -> bool:
        if primitive == "WalkForward":
            if not self.walk_client.wait_for_service(timeout_sec=1.0):
                return False
            req = Walk.Request()
            req.distance_meters = float(params.get('distance_m', 1.0))
            req.speed_mps = float(params.get('speed_mps', 0.3))
            future = self.walk_client.call_async(req)
            future.add_done_callback(lambda f: self._service_callback(f, "Walk"))
            return True

        elif primitive in ["WalkBackward", "WalkLeft", "WalkRight"]:
            return False

        elif primitive in ["RotateLeft", "RotateRight"]:
            if not self.rotate_client.wait_for_service(timeout_sec=1.0):
                return False
            req = Rotate.Request()
            angle_degrees = float(params.get('angle_degrees', 90.0))
            req.angle_degrees = angle_degrees if primitive == "RotateLeft" else -angle_degrees
            req.angular_speed_rps = float(params.get('angular_speed_rps', 0.6))
            future = self.rotate_client.call_async(req)
            future.add_done_callback(lambda f: self._service_callback(f, "Rotate"))
            return True

        elif primitive == "GraspHand":
            if not self.grasp_client.wait_for_service(timeout_sec=1.0):
                return False
            req = GraspHand.Request()
            req.strength = 1.0  # Default strength
            future = self.grasp_client.call_async(req)
            future.add_done_callback(lambda f: self._service_callback(f, "Grasp"))
            return True

        elif primitive == "ReleaseHand":
            if not self.release_client.wait_for_service(timeout_sec=1.0):
                return False
            req = ReleaseHand.Request()
            future = self.release_client.call_async(req)
            future.add_done_callback(lambda f: self._service_callback(f, "Release"))
            return True

        return False

    def _service_callback(self, future, service_name: str):
        try:
            response = future.result()
            if response.status.success:
                self.get_logger().info(f"{service_name} service succeeded: {response.status.message}")
            else:
                self.get_logger().warning(f"{service_name} service failed: {response.status.message}")
        except Exception as exc:
            self.get_logger().error(f"{service_name} service call failed: {exc}")

    def _execute_snippet(self, command: dict) -> None:
        snippet_name = command.get("snippet", {}).get("name", "")
        args = command.get("snippet", {}).get("args", {})
        executor = MOTOR_SCHEMA_EXECUTORS.get(snippet_name)
        if not executor:
            self.get_logger().warning(f"Unknown snippet '{snippet_name}', applying stop")
            self._stop_motion()
            return

        executor(self, args)

    def _start_linear_motion(self, linear_x: float, linear_y: float, distance_m: float, speed_mps: float) -> None:
        distance_m = max(0.05, min(abs(distance_m), 5.0))
        speed_mps = max(0.15, min(abs(speed_mps), 1.0))
        duration = (distance_m / speed_mps) * max(1.0, self.walk_duration_scale)
        self._start_motion(
            linear_x=linear_x,
            linear_y=linear_y,
            angular_z=0.0,
            duration_sec=duration,
            hold_heading=True,
        )

    def _start_rotation(self, angular_z: float, angle_degrees: float, angular_speed_rps: float) -> None:
        angle_rad = math.radians(max(5.0, min(abs(angle_degrees), 360.0)))
        angular_speed_rps = max(0.2, min(abs(angular_speed_rps), 1.5))
        duration = (angle_rad / angular_speed_rps) * max(1.0, self.rotation_duration_scale)
        self._start_motion(
            linear_x=0.0,
            linear_y=0.0,
            angular_z=angular_z,
            duration_sec=duration,
            hold_heading=False,
        )

    def _start_motion(
        self,
        *,
        linear_x: float,
        linear_y: float,
        angular_z: float,
        duration_sec: float,
        hold_heading: bool,
    ) -> None:
        self._warn_if_no_motion_subscribers()
        now = self.get_clock().now().nanoseconds / 1e9
        self.motion_end_sec = now + max(0.05, duration_sec)
        self.motion_linear_x = float(linear_x)
        self.motion_linear_y = float(linear_y)
        self.motion_angular_z = float(angular_z)
        self.motion_active = True
        self.heading_hold_enabled_for_motion = hold_heading
        self.target_yaw = self.current_yaw if hold_heading else None
        self._publish_twist(linear_x, linear_y, angular_z)

    def _start_walk_forward(self, distance_m: float, speed_mps: float) -> None:
        self._start_linear_motion(speed_mps, 0.0, distance_m, speed_mps)

    def _start_walk_backward(self, distance_m: float, speed_mps: float) -> None:
        self._start_linear_motion(-speed_mps, 0.0, distance_m, speed_mps)

    def _start_walk_left(self, distance_m: float, speed_mps: float) -> None:
        self._start_linear_motion(0.0, speed_mps, distance_m, speed_mps)

    def _start_walk_right(self, distance_m: float, speed_mps: float) -> None:
        self._start_linear_motion(0.0, -speed_mps, distance_m, speed_mps)

    def _start_rotate_left(self, angle_degrees: float, angular_speed_rps: float) -> None:
        self._start_rotation(abs(angular_speed_rps), angle_degrees, angular_speed_rps)

    def _start_rotate_right(self, angle_degrees: float, angular_speed_rps: float) -> None:
        self._start_rotation(-abs(angular_speed_rps), angle_degrees, angular_speed_rps)

    def _stop_motion(self) -> None:
        self.motion_active = False
        self.motion_linear_x = 0.0
        self.motion_linear_y = 0.0
        self.motion_angular_z = 0.0
        self.target_yaw = None
        self.heading_hold_enabled_for_motion = False
        for _ in range(3):
            self._publish_twist(0.0, 0.0, 0.0)

    def _motion_tick(self) -> None:
        if not self.motion_active:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now >= self.motion_end_sec:
            self._stop_motion()
            self.get_logger().info("Motion finished, robot stopped")
            self._start_next_pending_motion()
            return

        angular_z = self.motion_angular_z
        if (
            self.heading_hold_enabled_for_motion
            and self._should_apply_heading_hold()
            and self.current_yaw is not None
            and self.target_yaw is not None
        ):
            yaw_error = self._normalize_angle(self.target_yaw - self.current_yaw)
            correction = self.heading_hold_kp * yaw_error
            correction = max(
                -self.heading_hold_max_angular_z,
                min(self.heading_hold_max_angular_z, correction),
            )
            angular_z += correction

        self._publish_twist(self.motion_linear_x, self.motion_linear_y, angular_z)

    def _publish_twist(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.linear.y = float(linear_y)
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def _warn_if_no_motion_subscribers(self) -> None:
        if self.cmd_vel_pub.get_subscription_count() > 0:
            return
        self.get_logger().warning(
            f"No subscribers on motion topic {self.cmd_vel_topic}; command will not move the robot"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _should_apply_heading_hold(self) -> bool:
        if not self.enable_heading_hold:
            return False
        if self.heading_hold_use_sim_only and not self.is_sim_time:
            return False
        return True

    def _publish_outputs(self, command: dict) -> None:
        json_msg = String()
        json_msg.data = json.dumps(command, ensure_ascii=True)
        self.json_pub.publish(json_msg)

        status_msg = String()
        status_text = command.get("assistant_response", "")
        status_msg.data = status_text
        self.status_pub.publish(status_msg)
        self._remember_spoken_text(status_text)
        self._speak_text(status_text)

    @staticmethod
    def _get_pulse_env() -> dict:
        env = dict(os.environ)
        wslg_socket = "/mnt/wslg/runtime-dir/pulse/native"
        if not env.get("PULSE_SERVER") and os.path.exists(wslg_socket):
            env["PULSE_SERVER"] = f"unix:{wslg_socket}"
        return env

    def _speak_text(self, text: str) -> None:
        if not self.enable_tts:
            return
        text = (text or "").strip()
        if not text:
            return

        try:
            from gtts import gTTS

            tts = gTTS(text=text, lang=self.tts_language)
            tts.save(self.tts_tmp_file)

            try:
                result = subprocess.run(
                    [self.tts_player_cmd, self.tts_tmp_file],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._get_pulse_env(),
                    timeout=8,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"player returned {result.returncode}")
                return
            except Exception as exc:
                if not self._tts_warned:
                    self.get_logger().warning(f"Local TTS playback failed: {exc}")
                    self._tts_warned = True

            if self.auto_play_windows:
                self._play_on_windows(self.tts_tmp_file)
        except Exception as exc:
            self.get_logger().warning(f"TTS synth failed: {exc}")

    def _remember_spoken_text(self, text: str) -> None:
        normalized = self._normalize_text(text)
        if not normalized:
            return
        self.last_spoken_text = normalized
        self.last_spoken_time_sec = self.get_clock().now().nanoseconds / 1e9

    def _acknowledge_wake_word(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        self.awaiting_followup_until_sec = now + max(0.5, self.wake_followup_window_sec)
        self.get_logger().info("Wake word detected, waiting for follow-up")
        self._remember_spoken_text(self.wake_ack_text)
        self._speak_text(self.wake_ack_text)

    def _is_recent_tts_echo(self, raw_text: str) -> bool:
        recent_spoken = self.last_spoken_text
        if not recent_spoken:
            return False

        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_spoken_time_sec) > max(0.0, self.ignore_recent_tts_echo_sec):
            return False

        heard = self._normalize_text(raw_text)
        if not heard:
            return False

        return recent_spoken in heard or heard in recent_spoken

    def _extract_activated_text(self, raw_text: str) -> Optional[str]:
        cleaned = (raw_text or "").strip()
        if not cleaned:
            return None
        if not self.require_wake_word:
            return cleaned

        now = self.get_clock().now().nanoseconds / 1e9
        if now <= self.awaiting_followup_until_sec:
            self.awaiting_followup_until_sec = 0.0
            return cleaned

        normalized = self._normalize_text(cleaned)
        for wake_word in self.wake_words:
            if normalized == wake_word:
                return ""
            if normalized.startswith(f"{wake_word} "):
                pattern = rf"^\s*{re.escape(wake_word)}[\s,.:;!?-]*"
                activated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE).strip()
                return activated if activated else None
        return None

    def _play_on_windows(self, wsl_path: str) -> None:
        win_path = self._wsl_to_windows_path(wsl_path)
        if not win_path:
            return

        try:
            subprocess.run(
                ["cmd.exe", "/c", "start", "", win_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception as exc:
            self.get_logger().warning(f"Windows auto-play failed: {exc}")

        if not self.allow_windows_popup_fallback:
            return

        powershell = shutil.which("powershell.exe")
        if not powershell:
            return

        try:
            subprocess.run(
                [powershell, "-NoProfile", "-Command", f"Start-Process '{win_path}'"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    @staticmethod
    def _wsl_to_windows_path(wsl_path: str) -> str:
        prefix = "/mnt/"
        if not wsl_path.startswith(prefix) or len(wsl_path) < 7:
            return ""
        drive = wsl_path[5].upper()
        rest = wsl_path[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"

    @staticmethod
    def _extract_json(text: str) -> str:
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in AI response")
        return match.group(0)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _extract_number(text: str, default: float) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return default
        return float(match.group(1))

    @staticmethod
    def _extract_distance_m(text: str, default: float = 1.0) -> float:
        lowered = text.strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|meters)", lowered)
        if match:
            return float(match.group(1))

        english = {
            "half": 0.5,
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
        }
        for word, value in english.items():
            if re.search(rf"\b{word}\b", lowered):
                return value
        return default

    @staticmethod
    def _extract_angle_degrees(text: str, default: float = 90.0) -> float:
        lowered = text.strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(deg|degree|degrees)", lowered)
        if match:
            return float(match.group(1))
        if any(keyword in lowered for keyword in ["turn around"]):
            return 180.0
        if any(keyword in lowered for keyword in ["slight"]):
            return 30.0
        return default

    @staticmethod
    def _extract_duration_seconds(text: str, default: float = 0.0) -> float:
        lowered = text.strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(s|sec|second|seconds)\b", lowered)
        if match:
            return float(match.group(1))

        english = {
            "half second": 0.5,
            "one second": 1.0,
            "two seconds": 2.0,
            "three seconds": 3.0,
            "four seconds": 4.0,
            "five seconds": 5.0,
        }
        for phrase, value in english.items():
            if phrase in lowered:
                return value
        return default

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _canonical_primitive(name: str) -> str:
        key = name.strip().lower().replace("_", "")
        alias = {
            "walkforward": "WalkForward",
            "walkbackward": "WalkBackward",
            "walkleft": "WalkLeft",
            "walkright": "WalkRight",
            "stop": "Stop",
            "rotateleft": "RotateLeft",
            "rotateright": "RotateRight",
            "turnleft": "RotateLeft",
            "turnright": "RotateRight",
            "stand": "Stand",
            "sit": "Sit",
            "armunstow": "ArmUnstow",
            "armstow": "ArmStow",
            "grasphand": "GraspHand",
            "releasehand": "ReleaseHand",
            "extendarm": "ExtendArm",
        }
        return alias.get(key, name)

    @staticmethod
    def _category_for_primitive(primitive: str) -> str:
        if primitive in {"ArmUnstow", "ArmStow", "GraspHand", "ReleaseHand", "ExtendArm"}:
            return "manipulation"
        if primitive in {"Stand", "Sit"}:
            return "posture"
        return "locomotion"

    @staticmethod
    def _default_response_for_primitive(primitive: str) -> str:
        mapping = {
            "WalkForward": "Moving forward.",
            "WalkBackward": "Moving backward.",
            "WalkLeft": "Moving left.",
            "WalkRight": "Moving right.",
            "RotateLeft": "Turning left.",
            "RotateRight": "Turning right.",
            "Stand": "Standing by.",
            "Sit": "Sitting down.",
            "ArmUnstow": "Unstowing arm.",
            "ArmStow": "Stowing arm.",
            "GraspHand": "Closing gripper.",
            "ReleaseHand": "Opening gripper.",
            "Stop": "Stopping now.",
        }
        return mapping.get(primitive, "Executing command.")

    @staticmethod
    def _should_use_fast_path(raw_text: str) -> bool:
        text = raw_text.lower()
        low_latency_keywords = [
            "stop",
            "sit",
            "stand",
            "sit down",
            "stand up",
            "arm unstow",
            "arm stow",
            "unstow arm",
            "stow arm",
            "walk",
            "move",
            "forward",
            "backward",
            "left",
            "right",
            "turn",
            "rotate",
            "go forward",
            "go back",
            "go backward",
            "go left",
            "go right",
            "move forward",
            "move backward",
            "move left",
            "move right",
            "turn around",
            "close gripper",
            "open gripper",
            "close hand",
            "grip",
            "grasp hand",
            "open hand",
            "release",
            "release hand",
            "second",
            "seconds",
            "then",
            "and then",
        ]
        return any(keyword in text for keyword in low_latency_keywords)

    @staticmethod
    def _split_compound_command(raw_text: str) -> list[str]:
        text = (raw_text or "").strip()
        if not text:
            return []

        separators = [
            r"\band then\b",
            r"\bthen\b",
            r"\bafter that\b",
            r"\bnext\b",
        ]
        parts = [text]
        for pattern in separators:
            next_parts = []
            for part in parts:
                next_parts.extend(re.split(pattern, part, flags=re.IGNORECASE))
            parts = next_parts

        normalized = [part.strip(" ,.;") for part in parts if part.strip(" ,.;")]
        if len(normalized) > 1:
            return normalized

        # Comma-based splitting is much noisier with STT, so only allow it when
        # every candidate chunk still looks like a standalone motion command.
        comma_parts = [
            piece.strip(" ,.;")
            for piece in re.split(r"\s*,\s*", text)
            if piece.strip(" ,.;")
        ]
        if len(comma_parts) <= 1:
            return comma_parts

        if all(SpotVoiceAIPipeline._contains_motion_keyword(piece) for piece in comma_parts):
            return comma_parts

        return [text]

    def _start_next_pending_motion(self) -> None:
        if not self.pending_motion_commands:
            return
        next_command = self.pending_motion_commands.pop(0)
        primitive = next_command.get("behavior_execution", {}).get("primitive", "")
        self.get_logger().info(f"Starting queued motion command: {primitive}")
        self._execute_command(next_command, force_local=True)

    @staticmethod
    def _classify_primitive(lowered: str) -> str:
        english_text = re.sub(r"[^a-z\s]", " ", lowered)

        def contains_phrase(*phrases: str) -> bool:
            for phrase in phrases:
                if re.search(rf"\b{re.escape(phrase)}\b", english_text):
                    return True
            return False

        def contains_pattern(*patterns: str) -> bool:
            for pattern in patterns:
                if re.search(pattern, english_text):
                    return True
            return False

        if contains_phrase("stand up", "stand"):
            return "Stand"
        if contains_phrase("sit down", "sit", "seat"):
            return "Sit"
        if contains_phrase("arm unstow", "unstow arm", "unstow"):
            return "ArmUnstow"
        if contains_phrase("arm stow", "stow arm", "stow"):
            return "ArmStow"
        if contains_phrase("close gripper", "close hand", "grip", "grasp hand", "grasp", "grab"):
            return "GraspHand"
        if contains_phrase("open gripper", "open hand", "release hand", "release"):
            return "ReleaseHand"
        if contains_phrase("stop", "halt", "freeze"):
            return "Stop"
        if any(word in lowered for word in ["rotate left", "turn left", "turn around left"]):
            return "RotateLeft"
        if any(word in lowered for word in ["rotate right", "turn right", "turn around right"]):
            return "RotateRight"
        if contains_pattern(r"\b(move|go|walk)\s+(it\s+)?back(ward)?\b", r"\bback\s+up\b"):
            return "WalkBackward"
        if contains_pattern(
            r"\b(move|go|walk)\s+(it\s+)?(to\s+the\s+)?left\b",
            r"\b(left\s+step|step\s+left)\b",
        ):
            return "WalkLeft"
        if contains_pattern(
            r"\b(move|go|walk)\s+(it\s+)?(to\s+the\s+)?right\b",
            r"\b(right\s+step|step\s+right)\b",
        ):
            return "WalkRight"
        if contains_pattern(
            r"\b(move|go|walk)\s+(it\s+)?forward\b",
            r"\bgo\s+straight\b",
            r"\bforward\b",
        ):
            return "WalkForward"
        return "Stop"

    @staticmethod
    def _contains_motion_keyword(text: str) -> bool:
        lowered = text.lower()
        motion_keywords = [
            "stop",
            "halt",
            "freeze",
            "sit",
            "stand",
            "walk",
            "move",
            "forward",
            "backward",
            "left",
            "right",
            "turn",
            "rotate",
            "grasp",
            "grab",
            "open gripper",
            "close gripper",
            "open hand",
            "close hand",
            "grip",
            "release",
            "release hand",
            "stow",
            "unstow",
        ]
        return any(keyword in lowered for keyword in motion_keywords)

    @staticmethod
    def _contains_vision_keyword(text: str) -> bool:
        lowered = text.lower()
        vision_keywords = [
            "look",
            "vision",
            "see",
            "what do you see",
            "what can you see",
            "what is in front",
            "what's in front",
            "what objects",
            "detect",
            "camera",
            "object",
            "obstacle",
        ]
        return any(keyword in lowered for keyword in vision_keywords)

    @classmethod
    def _should_route_to_vision(cls, text: str) -> bool:
        # Safety-first rule: motion commands always stay in motion pipeline.
        if cls._contains_motion_keyword(text):
            return False
        return cls._contains_vision_keyword(text)

    @classmethod
    def _should_route_to_chat(cls, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False

        if cls._contains_motion_keyword(lowered):
            return False
        if cls._contains_vision_keyword(lowered):
            return False

        chat_keywords = [
            "hello",
            "hi",
            "hey",
            "how are you",
            "what is your name",
            "who are you",
            "thank you",
            "thanks",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(keyword in lowered for keyword in chat_keywords):
            return True

        return len(lowered.split()) >= 3


def main(args=None, parameter_defaults: Optional[dict] = None) -> None:
    rclpy.init(args=args)
    node = SpotVoiceAIPipeline(parameter_defaults=parameter_defaults)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
