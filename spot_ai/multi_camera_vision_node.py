#!/usr/bin/env python3

import os
from functools import partial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

try:
    import google.generativeai as genai
except Exception:
    genai = None


VISION_SYSTEM_PROMPT = (
    "You are Spot's multi-camera perception assistant. "
    "Use all provided camera views together. "
    "Describe what is around the robot, rough direction, and what nearby people or objects appear to be doing. "
    "Keep the answer concise: at most 3 short sentences."
)


class MultiCameraVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_multi_camera_vision_node")

        self.declare_parameter(
            "image_topics",
            [
                "/spot/camera/frontleft/image/compressed",
                "/spot/camera/frontright/image/compressed",
                "/spot/camera/left/image/compressed",
                "/spot/camera/right/image/compressed",
            ],
        )
        self.declare_parameter(
            "camera_labels",
            ["front-left", "front-right", "left", "right"],
        )
        self.declare_parameter("query_topic", "/spot_ai/vision_query")
        self.declare_parameter("output_topic", "/spot_ai/chat_output")
        self.declare_parameter(
            "default_query",
            "What is around you across all cameras?",
        )

        image_topics = self.get_parameter("image_topics").value
        camera_labels = self.get_parameter("camera_labels").value
        self.image_topics = [str(topic) for topic in image_topics]
        self.camera_labels = [str(label) for label in camera_labels]
        while len(self.camera_labels) < len(self.image_topics):
            self.camera_labels.append(f"camera-{len(self.camera_labels) + 1}")

        self.query_topic = str(self.get_parameter("query_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.default_query = str(self.get_parameter("default_query").value)

        self.latest_images = {}
        self.image_subs = []
        for idx, topic in enumerate(self.image_topics):
            label = self.camera_labels[idx]
            self.image_subs.append(
                self.create_subscription(
                    CompressedImage,
                    topic,
                    partial(self._on_image, label=label),
                    10,
                )
            )

        self.query_sub = self.create_subscription(String, self.query_topic, self._on_query, 10)
        self.output_pub = self.create_publisher(String, self.output_topic, 10)

        self.model = self._init_model()
        self.get_logger().info(
            f"multi_camera_vision_node ready: cameras={list(zip(self.camera_labels, self.image_topics))}"
        )

    def _init_model(self):
        api_key = os.getenv("GEMINI_API_KEY")
        preferred_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")

        if not genai:
            self.get_logger().warning("google.generativeai unavailable, multi-camera vision fallback enabled")
            return None
        if not api_key:
            self.get_logger().warning("GEMINI_API_KEY not set, multi-camera vision fallback enabled")
            return None

        try:
            genai.configure(api_key=api_key)
            for model_name in [preferred_model, "gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    model = genai.GenerativeModel(model_name)
                    self.get_logger().info(f"Gemini multi-camera vision model enabled: {model_name}")
                    return model
                except Exception:
                    continue
            raise RuntimeError("No usable Gemini model for multi-camera vision")
        except Exception as exc:
            self.get_logger().warning(f"Gemini multi-camera init failed: {exc}")
            return None

    def _on_image(self, msg: CompressedImage, *, label: str) -> None:
        fmt = (msg.format or "").lower()
        mime_type = "image/png" if "png" in fmt else "image/jpeg"
        self.latest_images[label] = {
            "mime_type": mime_type,
            "data": bytes(msg.data),
        }

    def _on_query(self, msg: String) -> None:
        query = (msg.data or "").strip() or self.default_query
        answer = self._generate_caption(query)

        out = String()
        out.data = answer
        self.output_pub.publish(out)
        self.get_logger().info(f"multi_camera_vision: {answer}")

    def _generate_caption(self, query: str) -> str:
        if not self.latest_images:
            return "I do not have any camera frames yet."
        if not self.model:
            return "Camera frames received, but Gemini vision is unavailable."

        content = [f"{VISION_SYSTEM_PROMPT}\n\nUser request: {query}\n"]
        for label in self.camera_labels:
            image = self.latest_images.get(label)
            if not image:
                continue
            content.append(f"Camera view: {label}")
            content.append(
                {
                    "mime_type": image["mime_type"],
                    "data": image["data"],
                }
            )

        if len(content) == 1:
            return "I do not have any recent camera frames yet."

        try:
            response = self.model.generate_content(content)
            text = (response.text or "").strip()
            return text or "I cannot confidently summarize the current surroundings."
        except Exception as exc:
            self.get_logger().warning(f"Multi-camera vision generation failed: {exc}")
            return "Multi-camera vision analysis failed."


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MultiCameraVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
