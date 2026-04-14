import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class WavInputWatcherNode(Node):
    def __init__(self) -> None:
        super().__init__("wav_input_watcher_node")

        self.declare_parameter(
            "audio_file", "/mnt/c/Users/jaeyk/Desktop/Spot/LLM-SPOT-Project/input.wav"
        )
        self.declare_parameter("whisper_model", "base")
        self.declare_parameter("whisper_language", "en")
        self.declare_parameter("whisper_task", "transcribe")
        self.declare_parameter("whisper_temperature", 0.0)
        self.declare_parameter("target_topic", "/spot_ai/voice_text")
        self.declare_parameter("publish_repeats", 1)
        self.declare_parameter("poll_interval_sec", 0.5)
        self.declare_parameter("settle_time_sec", 1.0)
        self.declare_parameter("min_file_size_bytes", 1024)

        self.audio_file = str(self.get_parameter("audio_file").value)
        self.whisper_model = str(self.get_parameter("whisper_model").value)
        self.whisper_language = str(self.get_parameter("whisper_language").value)
        self.whisper_task = str(self.get_parameter("whisper_task").value)
        self.whisper_temperature = float(self.get_parameter("whisper_temperature").value)
        self.target_topic = str(self.get_parameter("target_topic").value)
        self.publish_repeats = max(1, int(self.get_parameter("publish_repeats").value))
        self.poll_interval_sec = max(0.1, float(self.get_parameter("poll_interval_sec").value))
        self.settle_time_sec = max(0.0, float(self.get_parameter("settle_time_sec").value))
        self.min_file_size_bytes = max(1, int(self.get_parameter("min_file_size_bytes").value))

        self.pub = self.create_publisher(String, self.target_topic, 10)
        self.timer = self.create_timer(self.poll_interval_sec, self._poll_audio_file)

        self._whisper_model_instance = None
        self._seen_signature = None
        self._pending_signature = None
        self._pending_since_sec = 0.0

        self.get_logger().info(
            f"Watching WAV file for changes: {self.audio_file} -> {self.target_topic}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _stat_signature(self):
        if not os.path.exists(self.audio_file):
            return None
        try:
            stat = os.stat(self.audio_file)
        except OSError as exc:
            self.get_logger().warning(f"Failed to stat audio file: {exc}")
            return None
        if stat.st_size < self.min_file_size_bytes:
            return None
        return (int(stat.st_mtime_ns), int(stat.st_size))

    def _poll_audio_file(self) -> None:
        signature = self._stat_signature()
        if signature is None:
            return

        if signature == self._seen_signature:
            return

        now_sec = self._now_sec()
        if signature != self._pending_signature:
            self._pending_signature = signature
            self._pending_since_sec = now_sec
            return

        if (now_sec - self._pending_since_sec) < self.settle_time_sec:
            return

        self._transcribe_and_publish()
        self._seen_signature = signature
        self._pending_signature = None
        self._pending_since_sec = 0.0

    def _get_whisper_model(self):
        if self._whisper_model_instance is not None:
            return self._whisper_model_instance

        try:
            import whisper
        except Exception as exc:
            self.get_logger().error(
                "openai-whisper is not available. Install it before running this node. "
                f"Details: {exc}"
            )
            return None

        self.get_logger().info(f"Loading Whisper model '{self.whisper_model}'")
        self._whisper_model_instance = whisper.load_model(self.whisper_model)
        return self._whisper_model_instance

    def _transcribe_and_publish(self) -> None:
        model = self._get_whisper_model()
        if model is None:
            return

        self.get_logger().info(
            "Detected updated WAV file; transcribing "
            f"language='{self.whisper_language}' task='{self.whisper_task}': {self.audio_file}"
        )

        try:
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
            self.get_logger().warning("Transcription is empty. Nothing was published.")
            return

        msg = String()
        msg.data = text
        for _ in range(self.publish_repeats):
            self.pub.publish(msg)

        self.get_logger().info(f"Published text to {self.target_topic}: {text}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WavInputWatcherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
