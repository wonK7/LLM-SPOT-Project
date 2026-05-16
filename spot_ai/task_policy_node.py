#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from spot_schemas_interfaces.srv import GraspHand, ReleaseHand


class SpotTaskPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("spot_task_policy_node")

        self.declare_parameter("task_input_topic", "/spot_ai/task_input")
        self.declare_parameter("task_status_topic", "/spot_ai/task_status")
        self.declare_parameter("vision_query_topic", "/spot_ai/vision_query")

        task_input_topic = str(self.get_parameter("task_input_topic").value)
        task_status_topic = str(self.get_parameter("task_status_topic").value)
        vision_query_topic = str(self.get_parameter("vision_query_topic").value)

        self.task_sub = self.create_subscription(String, task_input_topic, self._on_task, 10)
        self.status_pub = self.create_publisher(String, task_status_topic, 10)
        self.vision_pub = self.create_publisher(String, vision_query_topic, 10)

        self.arm_unstow_client = self.create_client(Trigger, "/arm_unstow")
        self.arm_stow_client = self.create_client(Trigger, "/arm_stow")
        self.grasp_client = self.create_client(GraspHand, "/schemas/grasp_hand")
        self.release_client = self.create_client(ReleaseHand, "/schemas/release_hand")

        self.get_logger().info(
            f"task_policy_node ready: in={task_input_topic}, status={task_status_topic}, vision={vision_query_topic}"
        )

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _on_task(self, msg: String) -> None:
        raw_text = (msg.data or "").strip()
        if not raw_text:
            return

        lowered = raw_text.lower()

        if self._matches_inspect_scene(lowered):
            self._run_inspect_scene(raw_text)
            return

        if self._matches_prepare_grasp(lowered):
            self._run_prepare_grasp()
            return

        if self._matches_secure_object(lowered):
            self._run_secure_object()
            return

        self._publish_status(f"Unknown task: {raw_text}")

    @staticmethod
    def _matches_inspect_scene(text: str) -> bool:
        return any(keyword in text for keyword in [
            "look",
            "see",
            "vision",
            "inspect",
            "what is in front",
            "what objects",
            "obstacle",
        ])

    @staticmethod
    def _matches_prepare_grasp(text: str) -> bool:
        return any(keyword in text for keyword in [
            "prepare grasp",
            "prepare to grasp",
            "prepare to pick",
            "get ready to pick",
        ])

    @staticmethod
    def _matches_secure_object(text: str) -> bool:
        return any(keyword in text for keyword in [
            "secure object",
            "pick object",
            "grab object",
            "finish grasp",
        ])

    def _run_inspect_scene(self, raw_text: str) -> None:
        msg = String()
        msg.data = raw_text
        self.vision_pub.publish(msg)
        self._publish_status("Task matched: inspect_scene -> forwarded to vision query")

    def _run_prepare_grasp(self) -> None:
        if not self._call_trigger(self.arm_unstow_client, "/arm_unstow"):
            return
        if not self._call_release():
            return
        self._publish_status("Task matched: prepare_grasp -> arm unstow + open gripper")

    def _run_secure_object(self) -> None:
        if not self._call_grasp():
            return
        if not self._call_trigger(self.arm_stow_client, "/arm_stow"):
            return
        self._publish_status("Task matched: secure_object -> close gripper + arm stow")

    def _call_trigger(self, client, service_name: str) -> bool:
        if not client.wait_for_service(timeout_sec=1.0):
            self._publish_status(f"Task failed: {service_name} not available")
            return False

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self._publish_status(f"Task failed: timeout calling {service_name}")
            return False

        try:
            response = future.result()
        except Exception as exc:
            self._publish_status(f"Task failed: {service_name} call error: {exc}")
            return False

        if not response.success:
            self._publish_status(f"Task failed: {service_name} rejected: {response.message}")
            return False

        return True

    def _call_release(self) -> bool:
        if not self.release_client.wait_for_service(timeout_sec=1.0):
            self._publish_status("Task failed: /schemas/release_hand not available")
            return False

        future = self.release_client.call_async(ReleaseHand.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self._publish_status("Task failed: timeout calling /schemas/release_hand")
            return False

        try:
            response = future.result()
        except Exception as exc:
            self._publish_status(f"Task failed: release_hand call error: {exc}")
            return False

        if not response.status.success:
            self._publish_status(f"Task failed: release_hand rejected: {response.status.message}")
            return False

        return True

    def _call_grasp(self) -> bool:
        if not self.grasp_client.wait_for_service(timeout_sec=1.0):
            self._publish_status("Task failed: /schemas/grasp_hand not available")
            return False

        req = GraspHand.Request()
        req.strength = 1.0
        future = self.grasp_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self._publish_status("Task failed: timeout calling /schemas/grasp_hand")
            return False

        try:
            response = future.result()
        except Exception as exc:
            self._publish_status(f"Task failed: grasp_hand call error: {exc}")
            return False

        if not response.status.success:
            self._publish_status(f"Task failed: grasp_hand rejected: {response.status.message}")
            return False

        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpotTaskPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
