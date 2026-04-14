#!/usr/bin/env python3

import os
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    import google.generativeai as genai
except Exception:
    genai = None


VISION_SYSTEM_PROMPT = (
    "You are Spot's object narration assistant. "
    "Describe only visible objects and rough relative position. "
    "Keep output concise: max 2 short sentences."
)


class SpotVisionCaptionNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_vision_caption_node")

        self.declare_parameter("image_topic", "/spot/camera/frontleft/image/compressed")
        self.declare_parameter("query_topic", "/spot_ai/vision_query")
        self.declare_parameter("output_topic", "/spot_ai/chat_output")
        self.declare_parameter("default_query", "What objects are in front of you?")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.query_topic = str(self.get_parameter("query_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.default_query = str(self.get_parameter("default_query").value)

        self.latest_image_bytes: Optional[bytes] = None
        self.latest_mime_type: str = "image/jpeg"

        self.image_sub = self.create_subscription(
            CompressedImage, self.image_topic, self._on_image, 10
        )
        self.query_sub = self.create_subscription(String, self.query_topic, self._on_query, 10)
        self.output_pub = self.create_publisher(String, self.output_topic, 10)

        self.model = self._init_model()
        self.get_logger().info(
            f"vision_caption_node ready: image={self.image_topic}, query={self.query_topic}, out={self.output_topic}"
        )

    def _init_model(self):
        api_key = os.getenv("GEMINI_API_KEY")
        preferred_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

        if not genai:
            self.get_logger().warning("google.generativeai unavailable, vision fallback enabled")
            return None
        if not api_key:
            self.get_logger().warning("GEMINI_API_KEY not set, vision fallback enabled")
            return None

        try:
            genai.configure(api_key=api_key)
            for model_name in [preferred_model, "gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.get_logger().info(f"Gemini vision model enabled: {model_name}")
                    return model
                except Exception:
                    continue
            raise RuntimeError("No usable Gemini model for vision")
        except Exception as exc:
            self.get_logger().warning(f"Gemini vision init failed, fallback enabled: {exc}")
            return None

    def _on_image(self, msg: CompressedImage) -> None:
        self.latest_image_bytes = bytes(msg.data)
        fmt = (msg.format or "").lower()
        if "png" in fmt:
            self.latest_mime_type = "image/png"
        else:
            self.latest_mime_type = "image/jpeg"

    def _on_query(self, msg: String) -> None:
        query = (msg.data or "").strip() or self.default_query
        answer = self._generate_caption(query)

        out = String()
        out.data = answer
        self.output_pub.publish(out)
        self.get_logger().info(f"vision_caption: {answer}")

    def _generate_caption(self, query: str) -> str:
        if self.latest_image_bytes is None:
            return "I do not have a camera frame yet."

        if not self.model:
            return "Camera frame received, but Gemini vision is unavailable."

        try:
            prompt = f"{VISION_SYSTEM_PROMPT}\n\nUser request: {query}\nAssistant:"
            content = [
                prompt,
                {
                    "mime_type": self.latest_mime_type,
                    "data": self.latest_image_bytes,
                },
            ]
            response = self.model.generate_content(content)
            text = (response.text or "").strip()
            if not text:
                return "I cannot confidently identify an object in this frame."
            return text
        except Exception as exc:
            self.get_logger().warning(f"Vision generation failed: {exc}")
            return "Vision analysis failed for this frame."


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpotVisionCaptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
