from .common import coerce_float


def execute(node, args: dict) -> None:
    angle_degrees = abs(coerce_float(args, "angle_degrees", 90.0))
    angular_speed_rps = abs(coerce_float(args, "angular_speed_rps", 0.6))
    node.get_logger().info(
        f"RotateLeft command received: angle={angle_degrees:.1f} deg, speed={angular_speed_rps:.2f} rad/s"
    )
    node._start_rotate_left(angle_degrees, angular_speed_rps)
