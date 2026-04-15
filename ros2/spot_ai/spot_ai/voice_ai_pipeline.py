import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from spot_ai.snippets import SNIPPET_EXECUTORS
from spot_ai.spot_schema_names import (
    PRIMITIVE_TO_SNIPPET,
    SCHEMA_COMMANDS,
    SUPPORTED_EXECUTION_SNIPPETS,
)

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
- GraspHand: Close gripper
- ReleaseHand: Open gripper
- ExtendArm: Extend arm by x,y,z target pose

Response style: assistant_response must be short (max 1 sentence, <= 40 chars).
Return JSON only. No markdown.
""".strip()


class SpotVoiceAIPipeline(Node):
    def __init__(self) -> None:
        super().__init__("spot_voice_ai_pipeline")

        self.declare_parameter("enable_tts", True)
        self.declare_parameter("tts_language", "en")
        self.declare_parameter("tts_player_cmd", "mpg123")
        self.declare_parameter("walk_duration_scale", 1.0)
        self.declare_parameter("sim_walk_duration_scale", 8.0)
        self.declare_parameter("auto_sim_scale", True)
        self.declare_parameter("allow_windows_popup_fallback", False)
        self.declare_parameter("enable_heading_hold", True)
        self.declare_parameter("heading_hold_use_sim_only", True)
        self.declare_parameter("heading_hold_kp", 1.2)
        self.declare_parameter("heading_hold_max_angular_z", 0.35)
        self.declare_parameter("odom_topic", "/Spot/odometry")
        self.declare_parameter(
            "tts_tmp_file", "/mnt/c/Users/jaeyk/Desktop/Spot/LLM-SPOT-Project/speech.mp3"
        )
        self.declare_parameter("auto_play_windows", True)

        self.voice_sub = self.create_subscription(
            String, "/spot_ai/voice_text", self._voice_text_cb, 10
        )
        self.json_pub = self.create_publisher(String, "/spot_ai/command_json", 10)
        self.status_pub = self.create_publisher(String, "/spot_ai/status", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.motion_timer = self.create_timer(0.1, self._motion_tick)
        self.motion_active = False
        self.motion_end_sec = 0.0
        self.current_speed = 0.0
        self.current_yaw = None
        self.target_yaw = None
        self._tts_warned = False

        self.enable_tts = self.get_parameter("enable_tts").value
        self.tts_language = self.get_parameter("tts_language").value
        self.tts_player_cmd = self.get_parameter("tts_player_cmd").value
        base_scale = float(self.get_parameter("walk_duration_scale").value)
        sim_scale = float(self.get_parameter("sim_walk_duration_scale").value)
        auto_sim_scale = bool(self.get_parameter("auto_sim_scale").value)
        use_sim_time = (
            bool(self.get_parameter("use_sim_time").value)
            if self.has_parameter("use_sim_time")
            else False
        )
        self.is_sim_time = use_sim_time
        self.walk_duration_scale = (
            sim_scale if (auto_sim_scale and use_sim_time) else base_scale
        )
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
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.tts_tmp_file = self.get_parameter("tts_tmp_file").value
        self.auto_play_windows = self.get_parameter("auto_play_windows").value

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, 10
        )

        self.model = self._init_model()
        self.get_logger().info(
            f"Motion scale active: {self.walk_duration_scale:.2f} (use_sim_time={use_sim_time})"
        )
        self.get_logger().info("spot_voice_ai_pipeline ready: /spot_ai/voice_text")

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

        self.get_logger().info(f"Input: {raw_text}")
        command = self._build_command_json(raw_text)
        self._execute_snippet(command)
        self._publish_outputs(command)

    def _build_command_json(self, raw_text: str) -> dict:
        fallback = self._fallback_parse(raw_text)
        if self._should_use_fast_path(raw_text):
            self.get_logger().info("Fast-path parser used for low-latency locomotion command")
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
        text = raw_text.lower()
        is_stop = "stop" in text
        is_forward = any(k in text for k in ["forward", "walk"])

        distance = max(0.1, min(self._extract_number(text, default=1.0), 5.0))
        speed = 0.7 if "fast" in text else 0.45

        if is_stop and not is_forward:
            primitive = "Stop"
            snippet = "stop"
            distance = 0.0
            speed = 0.0
            answer = "Stopping now."
        else:
            primitive = "WalkForward"
            snippet = "walk_forward"
            answer = "Moving forward."

        now = datetime.now(timezone.utc).isoformat()
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
                    "category": "locomotion",
                    "name": primitive,
                    "confidence": 0.7,
                },
            },
            "behavior_execution": {
                "primitive": primitive,
                "priority": "normal",
                "parameters": {
                    "distance_m": distance,
                    "speed_mps": speed,
                },
            },
            "snippet": {
                "name": snippet,
                "args": {
                    "distance_m": distance,
                    "speed_mps": speed,
                },
            },
            "assistant_response": answer,
        }

    def _normalize_command(self, data: dict, raw_text: str) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        inferred_distance = self._extract_distance_m(raw_text, default=1.0)

        primitive_raw = (
            data.get("behavior_execution", {})
            .get(
                "primitive",
                data.get("perception", {}).get("interpretation", {}).get("name", "WalkForward"),
            )
        )
        primitive = self._canonical_primitive(str(primitive_raw))
        if primitive not in SCHEMA_COMMANDS:
            primitive = "Stop"

        params = data.get("behavior_execution", {}).get("parameters", {})
        distance = float(params.get("distance_m", 1.0))
        speed = float(params.get("speed_mps", 0.45))
        angle = float(params.get("angle_degrees", 0.0))
        x = float(params.get("x", 0.0))
        y = float(params.get("y", 0.0))
        z = float(params.get("z", 0.0))

        if primitive == "Stop":
            distance = 0.0
            speed = 0.0
            angle = 0.0
        elif primitive == "WalkForward":
            if distance <= 0.11:
                distance = inferred_distance
            distance = max(0.1, min(distance, 5.0))
            speed = max(0.4, min(speed, 1.0))

        snippet_name = PRIMITIVE_TO_SNIPPET.get(primitive, "stop")
        category = "manipulation" if primitive in ["GraspHand", "ReleaseHand", "ExtendArm"] else "locomotion"
        short_text = str(data.get("assistant_response", "Executing command.")).strip()[:40] or "Executing command."

        return {
            "header": {
                "timestamp": data.get("header", {}).get("timestamp", now),
                "robot_id": data.get("header", {}).get("robot_id", "spot-01"),
                "pipeline_version": data.get("header", {}).get("pipeline_version", "v1"),
            },
            "perception": {
                "input_source": data.get("perception", {}).get("input_source", "speech"),
                "raw_input": raw_text,
                "interpretation": {
                    "category": category,
                    "name": primitive,
                    "confidence": float(
                        data.get("perception", {}).get("interpretation", {}).get("confidence", 0.8)
                    ),
                },
            },
            "behavior_execution": {
                "primitive": primitive,
                "priority": data.get("behavior_execution", {}).get("priority", "normal"),
                "parameters": {
                    "distance_m": distance,
                    "speed_mps": speed,
                    "angle_degrees": angle,
                    "x": x,
                    "y": y,
                    "z": z,
                },
            },
            "snippet": {
                "name": snippet_name,
                "args": {
                    "distance_m": distance,
                    "speed_mps": speed,
                    "angle_degrees": angle,
                    "x": x,
                    "y": y,
                    "z": z,
                },
            },
            "assistant_response": short_text,
        }

    def _execute_snippet(self, command: dict) -> None:
        snippet_name = command.get("snippet", {}).get("name", "")
        args = command.get("snippet", {}).get("args", {})

        executor = SNIPPET_EXECUTORS.get(snippet_name)
        if executor:
            primitive = command.get("behavior_execution", {}).get("primitive", "")
            distance = command.get("behavior_execution", {}).get("parameters", {}).get("distance_m", 0.0)
            speed = command.get("behavior_execution", {}).get("parameters", {}).get("speed_mps", 0.0)
            self.get_logger().info(
                f"Executing primitive={primitive} distance_m={float(distance):.2f} speed_mps={float(speed):.2f}"
            )
            executor(self, args)
            return

        if snippet_name not in SUPPORTED_EXECUTION_SNIPPETS:
            self.get_logger().warning(
                f"Snippet parsed but not implemented yet: {snippet_name}. For safety, stop is applied."
            )
        else:
            self.get_logger().warning(f"Unknown snippet: {snippet_name}, forcing stop")
        self._stop_motion()

    def _start_walk_forward(self, distance_m: float, speed_mps: float) -> None:
        distance_m = max(0.1, min(distance_m, 5.0))
        speed_mps = max(0.2, min(speed_mps, 1.0))
        duration = (distance_m / speed_mps) * max(1.0, self.walk_duration_scale)

        now = self.get_clock().now().nanoseconds / 1e9
        self.motion_end_sec = now + duration
        self.current_speed = speed_mps
        self.motion_active = True
        self.target_yaw = self.current_yaw
        self._publish_twist(speed_mps)

    @staticmethod
    def _should_use_fast_path(raw_text: str) -> bool:
        text = raw_text.lower()
        keywords = ["stop", "walk", "forward", "go forward", "앞", "전진", "정지"]
        return any(keyword in text for keyword in keywords)

    def _stop_motion(self) -> None:
        self.motion_active = False
        self.current_speed = 0.0
        self.target_yaw = None
        for _ in range(3):
            self._publish_twist(0.0)

    def _motion_tick(self) -> None:
        if not self.motion_active:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now >= self.motion_end_sec:
            self._stop_motion()
            self.get_logger().info("WalkForward completed, auto-stop")
            return

        angular_z = 0.0
        if self._should_apply_heading_hold() and self.current_yaw is not None and self.target_yaw is not None:
            yaw_error = self._normalize_angle(self.target_yaw - self.current_yaw)
            angular_z = self.heading_hold_kp * yaw_error
            angular_z = max(-self.heading_hold_max_angular_z, min(self.heading_hold_max_angular_z, angular_z))

        self._publish_twist(self.current_speed, angular_z)

    def _publish_twist(self, linear_x: float, angular_z: float = 0.0) -> None:
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

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

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _publish_outputs(self, command: dict) -> None:
        json_msg = String()
        json_msg.data = json.dumps(command, ensure_ascii=True)
        self.json_pub.publish(json_msg)

        status_msg = String()
        status_msg.data = command.get("assistant_response", "")
        self.status_pub.publish(status_msg)
        self._speak_text(status_msg.data)

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

            gTTS(text=text, lang=self.tts_language).save(self.tts_tmp_file)
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
                    raise RuntimeError(f"player exited {result.returncode}")
            except (subprocess.TimeoutExpired, RuntimeError):
                self._try_windows_autoplay()
                if not self._tts_warned:
                    self.get_logger().warning("WSL audio playback failed; tried Windows auto-play fallback.")
                    self._tts_warned = True
        except Exception as exc:
            self._try_windows_autoplay()
            if not self._tts_warned:
                self.get_logger().warning(
                    "TTS playback failed. Install gTTS/mpg123 or disable TTS "
                    f"with enable_tts:=false. Details: {exc}"
                )
                self._tts_warned = True

    def _try_windows_autoplay(self) -> None:
        if not self.auto_play_windows:
            return
        win_path = self._wsl_to_windows_path(self.tts_tmp_file)
        if not win_path:
            return

        if self._try_windows_hidden_play(win_path):
            return
        if not self.allow_windows_popup_fallback:
            return

        try:
            cmd_path = shutil.which("cmd.exe") or "/mnt/c/Windows/System32/cmd.exe"
            subprocess.run(
                [cmd_path, "/c", "start", "", win_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return

    def _try_windows_hidden_play(self, win_path: str) -> bool:
        ps_path = shutil.which("powershell.exe") or "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        file_uri = "file:///" + win_path.replace("\\", "/")
        safe_uri = file_uri.replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName PresentationCore;"
            "$uri=New-Object -TypeName System.Uri -ArgumentList '" + safe_uri + "';"
            "$p=New-Object -TypeName System.Windows.Media.MediaPlayer;"
            "$p.Open($uri);"
            "Start-Sleep -Milliseconds 500;"
            "$p.Play();"
            "Start-Sleep -Seconds 8"
        )
        try:
            result = subprocess.run(
                [ps_path, "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", ps_script],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=12,
            )
            return result.returncode == 0
        except Exception:
            return False

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
    def _extract_number(text: str, default: float) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return default
        return float(match.group(1))

    @staticmethod
    def _extract_distance_m(text: str, default: float = 1.0) -> float:
        lowered = text.strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*(m|meter|meters|미터)", lowered)
        if match:
            return float(match.group(1))

        english = {"half": 0.5, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}
        for word, value in english.items():
            if re.search(rf"\b{word}\b", lowered):
                return value

        korean = {"반": 0.5, "한": 1.0, "한번": 1.0, "두": 2.0, "세": 3.0, "네": 4.0, "다섯": 5.0}
        if "미터" in lowered:
            for word, value in korean.items():
                if word in lowered:
                    return value

        return default

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
            "stand": "Stand",
            "sit": "Sit",
            "grasphand": "GraspHand",
            "releasehand": "ReleaseHand",
            "extendarm": "ExtendArm",
        }
        return alias.get(key, name)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpotVoiceAIPipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
