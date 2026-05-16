#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class SpotSafetyGateNode(Node):
    def __init__(self, parameter_defaults=None) -> None:
        super().__init__("spot_safety_gate")

        defaults = parameter_defaults or {}
        self.declare_parameter("input_cmd_topic", defaults.get("input_cmd_topic", "/spot_ai/cmd_vel_raw"))
        self.declare_parameter("output_cmd_topic", defaults.get("output_cmd_topic", "/cmd_vel"))
        self.declare_parameter("scan_topic", defaults.get("scan_topic", "/scan"))
        self.declare_parameter("enabled", defaults.get("enabled", True))
        self.declare_parameter("stop_distance_m", defaults.get("stop_distance_m", 0.6))
        self.declare_parameter("forward_fov_deg", defaults.get("forward_fov_deg", 60.0))
        self.declare_parameter("deadman_timeout_sec", defaults.get("deadman_timeout_sec", 0.35))
        self.declare_parameter("publish_hz", defaults.get("publish_hz", 20.0))

        input_cmd_topic = str(self.get_parameter("input_cmd_topic").value)
        output_cmd_topic = str(self.get_parameter("output_cmd_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.stop_distance_m = float(self.get_parameter("stop_distance_m").value)
        self.forward_fov_deg = float(self.get_parameter("forward_fov_deg").value)
        self.deadman_timeout_sec = float(self.get_parameter("deadman_timeout_sec").value)
        publish_hz = max(1.0, float(self.get_parameter("publish_hz").value))

        self.cmd_sub = self.create_subscription(
            Twist, input_cmd_topic, self._cmd_cb, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, 10
        )
        self.cmd_pub = self.create_publisher(Twist, output_cmd_topic, 10)

        self.timer = self.create_timer(1.0 / publish_hz, self._tick)

        self.last_cmd: Twist = Twist()
        self.last_cmd_sec: float = 0.0
        self.last_scan_sec: float = 0.0
        self.min_forward_distance_m: Optional[float] = None
        self.was_blocked = False

        self.get_logger().info(
            f"Safety gate ready: in={input_cmd_topic} out={output_cmd_topic} scan={scan_topic}"
        )
        self.get_logger().info(
            f"enabled={self.enabled} stop_distance_m={self.stop_distance_m:.2f} "
            f"fov_deg={self.forward_fov_deg:.1f} deadman={self.deadman_timeout_sec:.2f}s"
        )

    def _cmd_cb(self, msg: Twist) -> None:
        self.last_cmd = msg
        self.last_cmd_sec = self.get_clock().now().nanoseconds / 1e9

    def _scan_cb(self, msg: LaserScan) -> None:
        self.last_scan_sec = self.get_clock().now().nanoseconds / 1e9
        self.min_forward_distance_m = self._compute_forward_min_distance(msg)

    def _compute_forward_min_distance(self, msg: LaserScan) -> Optional[float]:
        if not msg.ranges:
            return None

        half_fov_rad = math.radians(max(1.0, self.forward_fov_deg) / 2.0)
        best = None

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue

            angle = msg.angle_min + (i * msg.angle_increment)
            if -half_fov_rad <= angle <= half_fov_rad:
                if best is None or r < best:
                    best = r

        return best

    def _is_deadman_expired(self, now_sec: float) -> bool:
        if self.last_cmd_sec <= 0.0:
            return True
        return (now_sec - self.last_cmd_sec) > self.deadman_timeout_sec

    def _is_obstacle_blocked(self) -> bool:
        if self.min_forward_distance_m is None:
            return False
        return self.min_forward_distance_m < self.stop_distance_m

    @staticmethod
    def _zero_twist() -> Twist:
        return Twist()

    def _tick(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9

        if not self.enabled:
            self.cmd_pub.publish(self.last_cmd)
            return

        deadman_expired = self._is_deadman_expired(now_sec)
        blocked = self._is_obstacle_blocked()

        if deadman_expired:
            self.cmd_pub.publish(self._zero_twist())
            return

        wants_forward = self.last_cmd.linear.x > 0.0
        if wants_forward and blocked:
            self.cmd_pub.publish(self._zero_twist())
            if not self.was_blocked:
                distance = self.min_forward_distance_m
                self.get_logger().warning(
                    f"Obstacle block: min_forward={distance:.2f}m < {self.stop_distance_m:.2f}m"
                )
            self.was_blocked = True
            return

        if self.was_blocked:
            self.get_logger().info("Obstacle cleared: forwarding cmd_vel")
            self.was_blocked = False

        self.cmd_pub.publish(self.last_cmd)


def main(args=None, parameter_defaults=None) -> None:
    rclpy.init(args=args)
    node = SpotSafetyGateNode(parameter_defaults=parameter_defaults)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
