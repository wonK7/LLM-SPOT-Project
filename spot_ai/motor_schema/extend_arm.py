from .common import not_implemented


def execute(node, args: dict) -> None:
    if not isinstance(args, dict):
        args = {}

    logger = node.get_logger()

    try:
        target_x = float(args.get("x", 0.0))
        target_y = float(args.get("y", 0.0))
        target_z = float(args.get("z", 0.0))
    except (TypeError, ValueError):
        target_x, target_y, target_z = 0.0, 0.0, 0.0

    # We keep the parsed target in the log so the perception/planning side can be checked.
    logger.info(
        f"ExtendArm command received: x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f}"
    )
    not_implemented(node, "extend_arm")
