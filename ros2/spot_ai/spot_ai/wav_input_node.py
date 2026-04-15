import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WavInputNode(Node):
    def __init__(self) -> None:
        super().__init__("wav_input_node")

        self.declare_parameter(
            "audio_file", "/mnt/c/Users/jaeyk/Desktop/Spot/LLM-SPOT-Project/input.wav"
        )
        self.declare_parameter("whisper_model", "base")
        self.declare_parameter("whisper_language", "en")
        self.declare_parameter("whisper_task", "transcribe")
        self.declare_parameter("whisper_temperature", 0.0)
        self.declare_parameter("target_topic", "/spot_ai/voice_text")
        self.declare_parameter("publish_repeats", 1)

        self.audio_file = self.get_parameter("audio_file").get_parameter_value().string_value
        self.whisper_model = (
            self.get_parameter("whisper_model").get_parameter_value().string_value
        )
        self.whisper_language = (
            self.get_parameter("whisper_language").get_parameter_value().string_value
        )
        self.whisper_task = (
            self.get_parameter("whisper_task").get_parameter_value().string_value
        )
        self.whisper_temperature = (
            self.get_parameter("whisper_temperature").get_parameter_value().double_value
        )
        self.target_topic = (
            self.get_parameter("target_topic").get_parameter_value().string_value
        )
        self.publish_repeats = (
            self.get_parameter("publish_repeats").get_parameter_value().integer_value
        )

        self.pub = self.create_publisher(String, self.target_topic, 10)

        self._startup_timer = self.create_timer(1.0, self._run_once)
        self.completed = False

    def _run_once(self) -> None:
        if self.completed:
            return
        self.completed = True
        self._startup_timer.cancel()

        if not os.path.exists(self.audio_file):
            self.get_logger().error(f"Audio file not found: {self.audio_file}")
            return

        try:
            import whisper
        except Exception as exc:
            self.get_logger().error(
                "openai-whisper is not available. Install it before running this node. "
                f"Details: {exc}"
            )
            return

        self.get_logger().info(
            "Transcribing file with Whisper "
            f"model='{self.whisper_model}', language='{self.whisper_language}', "
            f"task='{self.whisper_task}': {self.audio_file}"
        )

        try:
            model = whisper.load_model(self.whisper_model)
            result = model.transcribe(
                self.audio_file,
                language=self.whisper_language,
                task=self.whisper_task,
                temperature=float(self.whisper_temperature),
                fp16=False,
            )
            text = (result.get("text") or "").strip()
        except Exception as exc:
            self.get_logger().error(f"Failed to transcribe wav file: {exc}")
            return

        if not text:
            self.get_logger().error("Transcription is empty. Nothing to publish.")
            return

        msg = String()
        msg.data = text

        repeats = max(1, int(self.publish_repeats))
        for _ in range(repeats):
            self.pub.publish(msg)

        self.get_logger().info(f"Published text to {self.target_topic}: {text}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WavInputNode()

    while rclpy.ok() and not node.completed:
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
