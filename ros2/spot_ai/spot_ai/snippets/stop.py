def execute(node, args: dict) -> None:
    del args
    node._stop_motion()
    node.get_logger().info("Snippet executed: stop")
