from .common import coerce_float


def execute(node, args: dict) -> None:
    distance = coerce_float(args, "distance_m", 1.0)
    speed = coerce_float(args, "speed_mps", 0.45)
    node.get_logger().info(
        f"WalkForward command received: distance={distance:.2f} m, speed={speed:.2f} m/s"
    )
    node._start_walk_forward(distance, speed)
