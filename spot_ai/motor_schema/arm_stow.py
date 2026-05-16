def execute(node, args: dict) -> None:
    del args
    node.get_logger().info("ArmStow command received")
    if not node._call_trigger_service(node.arm_stow_client, "/arm_stow", "arm_stow"):
        node.get_logger().warning("Falling back because /arm_stow is unavailable")