import os
import subprocess
import shutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import google.generativeai as genai
except Exception:
    genai = None


CHAT_SYSTEM_PROMPT = (
    "You are Spot's demo chat assistant. "
    "Reply in concise, natural English. "
    "Keep each response under 2 short sentences."
)


class SpotChatTTSNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_chat_tts_node")

        self.declare_parameter("chat_input_topic", "/spot_ai/chat_input")
        self.declare_parameter("chat_output_topic", "/spot_ai/chat_output")
        self.declare_parameter("status_topic", "/spot_ai/status")
        self.declare_parameter(
            "tts_file_wsl", "/mnt/c/Users/jaeyk/Desktop/Spot/LLM-SPOT-Project/speech.mp3"
        )
        self.declare_parameter("tts_lang", "en")
        self.declare_parameter("tts_player_cmd", "mpg123")
        self.declare_parameter("auto_play_windows", True)

        self.chat_input_topic = self.get_parameter("chat_input_topic").value
        self.chat_output_topic = self.get_parameter("chat_output_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.tts_file_wsl = self.get_parameter("tts_file_wsl").value
        self.tts_lang = self.get_parameter("tts_lang").value
        self.tts_player_cmd = self.get_parameter("tts_player_cmd").value
        self.auto_play_windows = self.get_parameter("auto_play_windows").value

        self.chat_sub = self.create_subscription(
            String, self.chat_input_topic, self._on_chat_input, 10
        )
        self.chat_output_sub = self.create_subscription(
            String, self.chat_output_topic, self._on_chat_output, 10
        )
        self.chat_pub = self.create_publisher(String, self.chat_output_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self._last_self_output = ""

        self.model = self._init_model()
        self.get_logger().info(
            f"chat_tts_node ready: in={self.chat_input_topic}, out={self.chat_output_topic}"
        )

    def _init_model(self):
        api_key = os.getenv("GEMINI_API_KEY")
        preferred_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not genai:
            self.get_logger().warning("google.generativeai unavailable, chat fallback enabled")
            return None
        if not api_key:
            self.get_logger().warning("GEMINI_API_KEY not set, chat fallback enabled")
            return None

        try:
            genai.configure(api_key=api_key)
            for model_name in [preferred_model, "gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.get_logger().info(f"Gemini chat model enabled: {model_name}")
                    return model
                except Exception:
                    continue
            raise RuntimeError("No usable Gemini model for chat")
        except Exception as exc:
            self.get_logger().warning(f"Gemini init failed, chat fallback enabled: {exc}")
            return None

    def _on_chat_input(self, msg: String) -> None:
        user_text = (msg.data or "").strip()
        if not user_text:
            return

        response_text = self._generate_reply(user_text)
        out = String()
        out.data = response_text
        self._last_self_output = response_text
        self.chat_pub.publish(out)
        self.status_pub.publish(out)

        self.get_logger().info(f"chat_output: {response_text}")
        self._speak_and_play(response_text)

    def _on_chat_output(self, msg: String) -> None:
        text = (msg.data or "").strip()
        if not text:
            return
        if text == self._last_self_output:
            self._last_self_output = ""
            return
        self.get_logger().info(f"speaking external chat_output: {text}")
        self.status_pub.publish(msg)
        self._speak_and_play(text)

    def _generate_reply(self, user_text: str) -> str:
        if not self.model:
            return f"I heard: {user_text}."

        try:
            prompt = f"{CHAT_SYSTEM_PROMPT}\n\nUser: {user_text}\nAssistant:"
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            return text or "I am ready."
        except Exception as exc:
            self.get_logger().warning(f"Chat generation failed, fallback used: {exc}")
            return f"I heard: {user_text}."

    def _speak_and_play(self, text: str) -> None:
        try:
            from gtts import gTTS

            os.makedirs(os.path.dirname(self.tts_file_wsl), exist_ok=True)
            gTTS(text=text, lang=self.tts_lang).save(self.tts_file_wsl)
        except Exception as exc:
            self.get_logger().warning(f"TTS synth failed: {exc}")
            return

        try:
            result = subprocess.run(
                [self.tts_player_cmd, self.tts_file_wsl],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self._get_pulse_env(),
                timeout=8,
            )
            if result.returncode == 0:
                return
            self.get_logger().warning(
                f"Local TTS playback failed: {self.tts_player_cmd} returned {result.returncode}"
            )
        except Exception as exc:
            self.get_logger().warning(f"Local TTS playback failed: {exc}")

        if not self.auto_play_windows:
            return

        win_path = self._wsl_to_windows_path(self.tts_file_wsl)
        if not win_path:
            self.get_logger().warning("Failed to map WSL path to Windows path for auto-play")
            return

        try:
            subprocess.run(
                [self._windows_cmd(), "/c", "start", "", win_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self.get_logger().warning(f"Windows auto-play failed: {exc}")

    @staticmethod
    def _get_pulse_env() -> dict:
        env = dict(os.environ)
        wslg_socket = "/mnt/wslg/runtime-dir/pulse/native"
        if not env.get("PULSE_SERVER") and os.path.exists(wslg_socket):
            env["PULSE_SERVER"] = f"unix:{wslg_socket}"
        return env

    @staticmethod
    def _windows_cmd() -> str:
        return shutil.which("cmd.exe") or "cmd.exe"

    @staticmethod
    def _wsl_to_windows_path(wsl_path: str) -> str:
        # Convert /mnt/c/Users/... -> C:\Users\...
        prefix = "/mnt/"
        if not wsl_path.startswith(prefix) or len(wsl_path) < 7:
            return ""
        drive = wsl_path[5].upper()
        rest = wsl_path[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpotChatTTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
