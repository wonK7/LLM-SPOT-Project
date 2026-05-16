def execute(node, args: dict) -> None:
    del args
    node.get_logger().info("ArmUnstow command received")
    if not node._call_trigger_service(node.arm_unstow_client, "/arm_unstow", "arm_unstow"):
        node.get_logger().warning("Falling back because /arm_unstow is unavailable")